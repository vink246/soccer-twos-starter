"""Build RLlib PPO trainer config from the same YAML as training (checkpoint restore / watch)."""

from typing import Any, Dict, Tuple

from soccer_rl.algorithms.ppo.env_config import (
    build_multiagent_config,
    get_env_type,
    zero_opponent_policy,
)
from soccer_rl.algorithms.ppo.training.model_config import build_model_config

NUM_ENVS_PER_WORKER_TRAINING_DEFAULT = 3


def build_ppo_restore_configs(
    cfg: Dict[str, Any],
    base_port: int,
    *,
    force_single_env_per_worker: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Match train_ppo_team / render_match PPO config (no tune.run / callbacks).

    When force_single_env_per_worker is True, sets num_envs_per_worker to 1 for restore
    (avoids Unity worker_id collisions with fixed_unity_worker_id).
    """
    res_cfg = cfg.get("resources") or {}
    rllib_cfg = cfg.get("rllib") or {}
    env_cfg = cfg.get("env") or {}
    ma_cfg = cfg.get("multiagent") or {}

    num_envs_per_worker = res_cfg.get(
        "num_envs_per_worker", NUM_ENVS_PER_WORKER_TRAINING_DEFAULT
    )
    model_cfg = build_model_config(rllib_cfg)
    variation_name = env_cfg.get("variation", "team_vs_policy")
    is_multiagent = bool(env_cfg.get("multiagent", False))
    policy_mode = ma_cfg.get("policy_mode", "shared_all")
    use_reward_wrapper = bool((env_cfg.get("reward") or {}).get("enabled", False))
    reward_cfg = env_cfg.get("reward") or {}
    if reward_cfg and "team_map" not in reward_cfg:
        reward_cfg["team_map"] = {"team_0": [0, 1], "team_1": [2, 3]}

    base_env_config: Dict[str, Any] = {
        "num_envs_per_worker": num_envs_per_worker,
        "variation": get_env_type(variation_name),
        "multiagent": is_multiagent,
        "single_player": bool(env_cfg.get("single_player", True)),
        "flatten_branched": bool(env_cfg.get("flatten_branched", True)),
        "opponent_policy": zero_opponent_policy,
        "base_port": base_port,
    }
    if use_reward_wrapper:
        base_env_config["reward"] = reward_cfg

    ppo_config: Dict[str, Any] = {
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": rllib_cfg.get("log_level", "INFO"),
        "framework": rllib_cfg.get("framework", "torch"),
        "env": "Soccer",
        "env_config": base_env_config,
        "model": model_cfg,
        "rollout_fragment_length": rllib_cfg.get("rollout_fragment_length", 500),
        "train_batch_size": rllib_cfg.get("train_batch_size", 12000),
    }
    ppo_config.update({k: v for k, v in rllib_cfg.items() if k not in {"model"}})
    ppo_config["model"] = model_cfg
    ppo_config["env_config"] = base_env_config

    if is_multiagent:
        ppo_config["multiagent"] = build_multiagent_config(base_env_config, policy_mode)

    if force_single_env_per_worker:
        ppo_config["num_envs_per_worker"] = 1
        base_env_config["num_envs_per_worker"] = 1

    return base_env_config, ppo_config
