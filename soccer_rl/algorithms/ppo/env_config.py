"""
Env and multiagent helpers shared by PPO training and checkpoint playback.

Keeps eval scripts from importing the training entrypoint module.
"""

from typing import Any, Dict

from ray import tune

from soccer_twos import EnvType

from soccer_rl.common.training_utils import create_rllib_env


def zero_opponent_policy(*_):
    return 0


def apply_team_vs_opponent(
    base_env_config: Dict[str, Any],
    env_cfg: Dict[str, Any],
    *,
    is_team_vs: bool,
    self_play_enabled: bool,
    team_vs_self_play: Dict[str, Any] = None,
) -> None:
    """
    Mutate ``base_env_config`` for ``team_vs_policy`` only.

    - Self-play: attach ``team_vs_self_play``; omit ``opponent_policy`` (Unity default random
      until the training callback swaps in the learned policy).
    - Else ``env.team_vs_opponent``: ``random`` → omit ``opponent_policy`` (Unity samples
      actions); ``zero`` → ``zero_opponent_policy`` (constant action 0).
    """
    base_env_config.pop("opponent_policy", None)
    base_env_config.pop("team_vs_self_play", None)
    if not is_team_vs:
        return
    if self_play_enabled:
        base_env_config["team_vs_self_play"] = dict(team_vs_self_play or {})
        return
    mode = str(env_cfg.get("team_vs_opponent", "zero")).lower().strip()
    if mode == "random":
        return
    if mode == "zero":
        base_env_config["opponent_policy"] = zero_opponent_policy
        return
    print(f"Warning: unknown env.team_vs_opponent {mode!r}; using zero (action 0).")
    base_env_config["opponent_policy"] = zero_opponent_policy


def get_env_type(name: str):
    mapping = {
        "team_vs_policy": EnvType.team_vs_policy,
        "multiagent_team": EnvType.multiagent_team,
        "multiagent_player": EnvType.multiagent_player,
    }
    return mapping.get(name, EnvType.team_vs_policy)


def policy_id_for_agent(agent_id, mode: str):
    try:
        aid = int(agent_id)
    except Exception:
        aid = 0
    if mode == "per_player":
        return f"player_{aid}"
    if mode == "team_shared":
        return "team_0" if aid in (0, 1) else "team_1"
    return "default"


def build_multiagent_config(env_config: Dict[str, Any], policy_mode: str):
    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    if policy_mode == "per_player":
        policies = {
            "player_0": (None, obs_space, act_space, {}),
            "player_1": (None, obs_space, act_space, {}),
            "player_2": (None, obs_space, act_space, {}),
            "player_3": (None, obs_space, act_space, {}),
        }
    elif policy_mode == "team_shared":
        policies = {
            "team_0": (None, obs_space, act_space, {}),
            "team_1": (None, obs_space, act_space, {}),
        }
    else:
        policies = {"default": (None, obs_space, act_space, {})}

    return {
        "policies": policies,
        "policy_mapping_fn": tune.function(
            lambda agent_id, *args, **kwargs: policy_id_for_agent(agent_id, policy_mode)
        ),
        "policies_to_train": list(policies.keys()),
    }
