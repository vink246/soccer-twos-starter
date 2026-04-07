"""
Build soccer_twos.make(...) kwargs from experiment config.

Strips Ray-only keys. Resolves string policy presets using a Discrete action space
inferred from variation + single_player + flatten_branched.
"""

from typing import Any, Dict, List, Optional, Tuple

import gym
import soccer_twos
from soccer_twos import EnvType

from soccer_rl.dense_reward_wrapper import maybe_wrap_dense_reward
from soccer_rl.policy_checkpoint import build_actor_policy_from_spec
from soccer_rl.policy_presets import build_policy

# Not passed to soccer_twos.make (Ray RLlib legacy).
_IGNORE_MAKE_KEYS = frozenset({"num_envs_per_worker", "multiagent"})


def _coerce_variation(v: Any) -> Any:
    if isinstance(v, str):
        return getattr(EnvType, v)
    return v


def infer_discrete_action_size(
    variation: Any, single_player: bool, flatten_branched: bool
) -> int:
    """Flattened discrete sizes from environment-information.md."""
    if variation == EnvType.multiagent_team and flatten_branched:
        return 729
    if variation == EnvType.team_vs_policy and single_player and flatten_branched:
        return 27
    if variation == EnvType.multiagent_player and flatten_branched:
        return 27
    if flatten_branched:
        return 27
    raise ValueError(
        "Cannot infer action size; set env.action_space_n in config for this variation."
    )


def apply_training_mode_to_env_section(
    env_section: Dict[str, Any], training_mode: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge training_mode flags into env kwargs for soccer_twos.make."""
    out = dict(env_section)
    tm = training_mode or {}
    single = tm.get("single_agent", True)
    if single:
        out["variation"] = _coerce_variation(out.get("variation", EnvType.team_vs_policy))
        out["single_player"] = True
        # Convenience: allow these to live under training_mode in YAML.
        for k in ("opponent_policy", "teammate_policy"):
            if k in tm and k not in out:
                out[k] = tm[k]
    else:
        out["variation"] = _coerce_variation(out.get("variation", EnvType.multiagent_team))
        out.pop("single_player", None)
    return out


def resolve_policy_value(
    val: Any, action_space: gym.spaces.Discrete
) -> Any:
    """String preset, checkpoint dict, or pass through callables / None."""
    if val is None or callable(val):
        return val
    if isinstance(val, str):
        return build_policy(val, action_space)
    if isinstance(val, dict):
        return build_actor_policy_from_spec(val, action_space)
    raise TypeError(
        "policy must be str, dict (checkpoint spec), callable, or None, got %s"
        % type(val)
    )


def _resolve_string_policies(
    cfg: Dict[str, Any],
    action_space: gym.spaces.Discrete,
) -> None:
    for key in ("opponent_policy", "teammate_policy"):
        if key in cfg:
            cfg[key] = resolve_policy_value(cfg[key], action_space)


def build_make_kwargs(
    full_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], gym.spaces.Discrete]:
    """
    From merged experiment config (YAML), produce kwargs for soccer_twos.make.

    Returns (make_kwargs, inferred_discrete_action_space).
    """
    env_section = dict(full_config.get("env") or {})
    training_mode = full_config.get("training_mode") or {}
    merged = apply_training_mode_to_env_section(env_section, training_mode)

    merged.setdefault("flatten_branched", True)
    merged["variation"] = _coerce_variation(merged.get("variation"))

    variation = merged["variation"]
    single_player = bool(merged.get("single_player", False))
    flatten = bool(merged.get("flatten_branched", True))

    if "action_space_n" in merged:
        n = int(merged.pop("action_space_n"))
    else:
        n = infer_discrete_action_size(variation, single_player, flatten)

    action_space = gym.spaces.Discrete(n)

    # Team opponent handled in trainer, not necessarily a make() kwarg.
    _resolve_string_policies(merged, action_space)

    for k in _IGNORE_MAKE_KEYS:
        merged.pop(k, None)

    return merged, action_space


def install_teammate_policy_on_env(env: Any, teammate_policy: Any) -> None:
    """
    ``soccer_twos.make`` does not forward ``teammate_policy`` to ``TeamVsPolicyWrapper``;
    install it by walking the wrapper chain.
    """
    if teammate_policy is None or not callable(teammate_policy):
        return
    cur: Any = env
    while cur is not None:
        setter = getattr(cur, "set_teammate_policy", None)
        if callable(setter):
            setter(teammate_policy)
            return
        cur = getattr(cur, "env", None)


def _soccer_twos_make_with_additional_args(
    env_config: Dict[str, Any], additional_args: List[str]
) -> Any:
    """
    Same launch sequence as soccer_twos.make, but forwards additional_args to UnityEnvironment
    (e.g. ``-batchmode`` without ``-nographics`` for windowless-but-rendered capture).
    """
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )
    from soccer_twos.package import ROLLOUT_ENV_PATH, TRAINING_ENV_PATH
    from soccer_twos.side_channels import EnvConfigurationChannel
    from soccer_twos.wrappers import (
        EnvChannelWrapper,
        EnvType,
        MultiAgentUnityWrapper,
        MultiagentTeamWrapper,
        TeamVsPolicyWrapper,
    )

    cfg = dict(env_config)
    if not cfg.get("env_path"):
        if cfg.get("watch"):
            cfg["env_path"] = ROLLOUT_ENV_PATH
            cfg["time_scale"] = 1
            cfg["quality_level"] = 5
            cfg["render"] = True
        else:
            cfg["env_path"] = TRAINING_ENV_PATH

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(
        time_scale=cfg.get("time_scale", 20),
        quality_level=cfg.get("quality_level", 0),
        target_frame_rate=cfg.get("target_frame_rate"),
        capture_frame_rate=cfg.get("capture_frame_rate"),
    )
    if cfg.get("env_channel"):
        env_channel = cfg.get("env_channel")
    else:
        env_channel = EnvConfigurationChannel()
    env_channel.set_parameters(
        blue_team_name=cfg.get("blue_team_name"),
        orange_team_name=cfg.get("orange_team_name"),
    )

    unity_env = UnityEnvironment(
        cfg["env_path"],
        no_graphics=not cfg.get("render", False),
        base_port=cfg.get("base_port", 50039),
        worker_id=cfg.get("worker_id", 0),
        side_channels=[engine_channel, env_channel],
        additional_args=list(additional_args),
    )

    multiagent_config = {
        k: cfg[k]
        for k in [
            "uint8_visual",
            "flatten_branched",
            "action_space_seed",
            "termination_mode",
        ]
        if k in cfg
    }
    env = MultiAgentUnityWrapper(unity_env, **multiagent_config)

    if "variation" in cfg:
        if EnvType(cfg["variation"]) is EnvType.multiagent_player:
            pass
        elif EnvType(cfg["variation"]) is EnvType.multiagent_team:
            env = MultiagentTeamWrapper(env)
        elif EnvType(cfg["variation"]) is EnvType.team_vs_policy:
            env = TeamVsPolicyWrapper(
                env,
                opponent_policy=cfg["opponent_policy"]
                if "opponent_policy" in cfg
                else None,
                teammate_policy=cfg["teammate_policy"]
                if "teammate_policy" in cfg
                else None,
                single_player=cfg["single_player"] if "single_player" in cfg else False,
            )
        else:
            raise ValueError(
                "Variation parameter invalid. Must be an EnvType member: "
                + str([e.value for e in EnvType])
                + ". Received "
                + str(cfg["variation"])
            )

    env = EnvChannelWrapper(env, env_channel)
    return env


def make_env(full_config: Dict[str, Any]):
    """Instantiate the Unity-backed env from a full experiment config dict."""
    kwargs, _ = build_make_kwargs(full_config)
    uargs: List[str] = []
    raw_extra = kwargs.pop("unity_additional_args", None)
    if raw_extra is not None:
        uargs.extend(list(raw_extra))
    if kwargs.pop("video_no_window", False):
        uargs = ["-batchmode"] + uargs
    needs_custom_make = bool(uargs) or kwargs.get("target_frame_rate") is not None or (
        kwargs.get("capture_frame_rate") is not None
    )
    teammate = kwargs.get("teammate_policy")
    if needs_custom_make:
        env = _soccer_twos_make_with_additional_args(kwargs, uargs)
    else:
        env = soccer_twos.make(**kwargs)
        install_teammate_policy_on_env(env, teammate)
    return maybe_wrap_dense_reward(env, full_config)


def normalize_worker_env_config(env_config: Any) -> Dict[str, Any]:
    """Turn RLlib-style env config (or dict) into a plain dict with worker_id set."""
    if hasattr(env_config, "worker_index"):
        d = env_config.copy() if hasattr(env_config, "copy") else dict(env_config)
        d["worker_id"] = (
            env_config.worker_index * d.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
        return d
    return dict(env_config)


def make_env_from_flat_config(env_config: Dict[str, Any]):
    """
    Build env from a flat config dict (e.g. Ray worker). Resolves string policies
    using inferred action space; strips Ray-only keys.
    """
    cfg = normalize_worker_env_config(env_config)
    cfg = dict(cfg)
    for k in _IGNORE_MAKE_KEYS:
        cfg.pop(k, None)

    cfg.setdefault("flatten_branched", True)
    cfg["variation"] = _coerce_variation(cfg.get("variation", EnvType.multiagent_player))

    variation = cfg["variation"]
    single_player = bool(cfg.get("single_player", False))
    flatten = bool(cfg.get("flatten_branched", True))
    if "action_space_n" in cfg:
        n = int(cfg.pop("action_space_n"))
    else:
        n = infer_discrete_action_size(variation, single_player, flatten)
    space = gym.spaces.Discrete(n)
    _resolve_string_policies(cfg, space)
    if cfg.get("target_frame_rate") is not None or cfg.get("capture_frame_rate") is not None:
        env = _soccer_twos_make_with_additional_args(dict(cfg), [])
    else:
        env = soccer_twos.make(**cfg)
        install_teammate_policy_on_env(env, cfg.get("teammate_policy"))
    return maybe_wrap_dense_reward(env, cfg)
