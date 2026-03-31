"""
Team PPO training script for Soccer-Twos.

- Reads run and RLlib options from a YAML config (default: PPO/configs/config.yaml).
- CLI flags override config file values.
- Puts each run in its own timestamped folder under team_runs/ (or config run.output_dir).
- Disables the Ray dashboard (avoids hostname resolution issues on WSL/PACE).
- Saves learning-curve plots every N training iterations.

Usage:
  python PPO/training/train_ppo_team.py
  python PPO/training/train_ppo_team.py --config PPO/configs/config.yaml
  python PPO/training/train_ppo_team.py --max-timesteps 500000

Run from the repository root. On PACE ICE: set PACE_NUM_GPUS if needed.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Ensure repo root is on path so we can import training_utils
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_utils import (
    create_rllib_env,
    load_config,
    get_num_gpus,
    print_gpu_status,
    PlotCallback,
    ProgressPrintCallback,
    has_matplotlib,
)
from PPO.training.model_config import build_model_config
from PPO.training.goal_metrics_callbacks import GoalStatsCallbacks

import ray
from ray import tune
from soccer_twos import EnvType

# PPO-specific defaults
NUM_ENVS_PER_WORKER = 3
DEFAULT_PLOT_FREQ = 10
DEFAULT_MAX_TIMESTEPS = 2_000_000
DEFAULT_NUM_WORKERS = 8
BASE_OUTPUT_DIR = "team_runs"
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "..", "configs", "config.yaml")


def _zero_opponent_policy(*_):
    return 0


def _get_env_type(name: str):
    mapping = {
        "team_vs_policy": EnvType.team_vs_policy,
        "multiagent_team": EnvType.multiagent_team,
        "multiagent_player": EnvType.multiagent_player,
    }
    return mapping.get(name, EnvType.team_vs_policy)


def _policy_id_for_agent(agent_id, mode: str):
    try:
        aid = int(agent_id)
    except Exception:
        aid = 0
    if mode == "per_player":
        return f"player_{aid}"
    if mode == "team_shared":
        return "team_0" if aid in (0, 1) else "team_1"
    return "default"


def _build_multiagent_config(env_config: Dict[str, Any], policy_mode: str):
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
            lambda agent_id, *args, **kwargs: _policy_id_for_agent(agent_id, policy_mode)
        ),
        "policies_to_train": list(policies.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="Team PPO training with plots and run folders")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override run.output_dir from config")
    parser.add_argument("--plot-freq", type=int, default=None, help="Override run.plot_freq")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Override run.max_timesteps")
    parser.add_argument("--num-workers", type=int, default=None, help="Override resources.num_workers")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override resources.num_gpus")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_cfg = cfg.get("run") or {}
    res_cfg = cfg.get("resources") or {}
    rllib_cfg = cfg.get("rllib") or {}
    env_cfg = cfg.get("env") or {}
    ma_cfg = cfg.get("multiagent") or {}

    output_dir = args.output_dir or run_cfg.get("output_dir", BASE_OUTPUT_DIR)
    plot_freq = args.plot_freq if args.plot_freq is not None else run_cfg.get("plot_freq", DEFAULT_PLOT_FREQ)
    max_timesteps = args.max_timesteps if args.max_timesteps is not None else run_cfg.get("max_timesteps", DEFAULT_MAX_TIMESTEPS)
    num_workers = args.num_workers if args.num_workers is not None else res_cfg.get("num_workers", DEFAULT_NUM_WORKERS)
    num_envs_per_worker = res_cfg.get("num_envs_per_worker", NUM_ENVS_PER_WORKER)
    checkpoint_freq = run_cfg.get("checkpoint_freq", 100)

    num_gpus = args.num_gpus
    if num_gpus is None:
        from_cfg = res_cfg.get("num_gpus")
        num_gpus = get_num_gpus() if from_cfg is None else int(from_cfg)

    run_name = f"PPO_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Config: {args.config}")
    print(f"Run directory: {os.path.abspath(run_dir)}")
    print(f"Max timesteps: {max_timesteps:,}  (stop when total env steps reach this)")
    print(f"Workers: {num_workers}  |  GPUs: {num_gpus}  |  Plot every: {plot_freq} iters")
    print_gpu_status(num_gpus)
    if not has_matplotlib():
        print("Warning: matplotlib not installed; plots will not be saved. pip install matplotlib")

    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    plot_callback = PlotCallback(plot_freq=plot_freq)
    progress_callback = ProgressPrintCallback(max_timesteps=max_timesteps)

    model_cfg = build_model_config(rllib_cfg)
    variation_name = env_cfg.get("variation", "team_vs_policy")
    is_multiagent = bool(env_cfg.get("multiagent", False))
    policy_mode = ma_cfg.get("policy_mode", "shared_all")
    use_reward_wrapper = bool((env_cfg.get("reward") or {}).get("enabled", False))
    reward_cfg = env_cfg.get("reward") or {}
    if reward_cfg and "team_map" not in reward_cfg:
        reward_cfg["team_map"] = {"team_0": [0, 1], "team_1": [2, 3]}

    base_env_config = {
        "num_envs_per_worker": num_envs_per_worker,
        "variation": _get_env_type(variation_name),
        "multiagent": is_multiagent,
        "single_player": bool(env_cfg.get("single_player", True)),
        "flatten_branched": bool(env_cfg.get("flatten_branched", True)),
        "opponent_policy": _zero_opponent_policy,
    }
    if use_reward_wrapper:
        base_env_config["reward"] = reward_cfg

    ppo_config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
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
        ppo_config["multiagent"] = _build_multiagent_config(base_env_config, policy_mode)

    # RLlib-side callbacks for extra metrics (goals_for / goals_against).
    ppo_config["callbacks"] = GoalStatsCallbacks

    print(
        "Env mode:",
        variation_name,
        "| multiagent:",
        is_multiagent,
        "| policy_mode:",
        policy_mode if is_multiagent else "single_policy",
    )
    print("Reward shaping:", "enabled" if use_reward_wrapper else "disabled")

    analysis = tune.run(
        "PPO",
        name="PPO_Soccer",
        config=ppo_config,
        stop={"timesteps_total": max_timesteps},
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=run_dir,
        callbacks=[plot_callback, progress_callback],
        verbose=0,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial:
        print("Best trial:", best_trial)
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print("Best checkpoint:", best_checkpoint)
    print("Done. Results in:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
