"""
Render Soccer-Twos PPO matches to video from checkpoints and/or random agents.

Restore requires the same RLlib config as training (model, batch sizes, multiagent).
Pass --config pointing to the same YAML you used for training.

Examples:
  python PPO/eval/render_match.py --config PPO/configs/config.yaml \\
    --team-a-checkpoint <ckptA> --team-b-strategy random --headless
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PPO.training.model_config import build_model_config  # noqa: E402
from PPO.training.train_ppo_team import (  # noqa: E402
    _build_multiagent_config,
    _get_env_type,
    _zero_opponent_policy,
)
from training_utils import create_rllib_env, load_config  # noqa: E402

NUM_ENVS_PER_WORKER = 3
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "..", "configs", "config.yaml")


def _start_virtual_display_if_needed(headless: bool, display: str, size: str):
    if not headless:
        return None
    xvfb_path = shutil.which("Xvfb")
    if xvfb_path is None:
        raise RuntimeError(
            "Headless video requested but Xvfb was not found on PATH. "
            "Install Xvfb or run with xvfb-run."
        )
    os.environ.setdefault("DISPLAY", display)
    cmd = [xvfb_path, os.environ["DISPLAY"], "-screen", "0", size, "-nolisten", "tcp"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return proc


def _goal_scored(done, reward, goal_threshold: float = 0.9) -> bool:
    if isinstance(done, dict) and done.get("__all__", False):
        if isinstance(reward, dict):
            return any(abs(float(v)) >= goal_threshold for v in reward.values())
        return abs(float(reward)) >= goal_threshold
    return False


def _assemble_base_env_and_ppo_config(
    cfg: Dict[str, Any],
    base_port: int,
) -> tuple:
    """
    Match train_ppo_team.py PPO config (minus callbacks / tune.run).
    """
    res_cfg = cfg.get("resources") or {}
    rllib_cfg = cfg.get("rllib") or {}
    env_cfg = cfg.get("env") or {}
    ma_cfg = cfg.get("multiagent") or {}

    num_envs_per_worker = res_cfg.get("num_envs_per_worker", NUM_ENVS_PER_WORKER)
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
        "variation": _get_env_type(variation_name),
        "multiagent": is_multiagent,
        "single_player": bool(env_cfg.get("single_player", True)),
        "flatten_branched": bool(env_cfg.get("flatten_branched", True)),
        "opponent_policy": _zero_opponent_policy,
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
        ppo_config["multiagent"] = _build_multiagent_config(base_env_config, policy_mode)

    return base_env_config, ppo_config


def _build_trainer(checkpoint: str, ppo_config: Dict[str, Any]) -> PPOTrainer:
    trainer = PPOTrainer(config=ppo_config)
    trainer.restore(checkpoint)
    return trainer


def _get_action(
    agent_id: int,
    obs,
    team: str,
    team_strategies: Dict[str, str],
    team_trainers: Dict[str, Optional[PPOTrainer]],
    team_policy_ids: Dict[str, str],
):
    strategy = team_strategies[team]
    if strategy == "random":
        return None
    trainer = team_trainers.get(team)
    if trainer is None:
        return None
    policy_id = team_policy_ids.get(team) or "default"
    action = trainer.compute_single_action(obs, policy_id=policy_id)
    return action


def main():
    parser = argparse.ArgumentParser(description="Render PPO checkpoint matches to video")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Same YAML as training (required for checkpoint restore to match model/env)",
    )
    parser.add_argument("--team-a-checkpoint", type=str, default=None)
    parser.add_argument("--team-b-checkpoint", type=str, default=None)
    parser.add_argument("--team-a-strategy", type=str, default="checkpoint", choices=["checkpoint", "random"])
    parser.add_argument("--team-b-strategy", type=str, default="checkpoint", choices=["checkpoint", "random"])
    parser.add_argument("--team-a-policy-id", type=str, default="team_0")
    parser.add_argument("--team-b-policy-id", type=str, default="team_1")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default=None,
        choices=["team_shared", "per_player", "shared_all"],
        help="Override multiagent.policy_mode from YAML if set",
    )
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-video", type=str, default="PPO/runs/playback.mp4")
    parser.add_argument("--output-trace", type=str, default="PPO/runs/playback_trace.json")
    parser.add_argument("--headless", action="store_true", help="Use Xvfb and do not require an attached display")
    parser.add_argument("--display", type=str, default=":99", help="Display to use for headless mode")
    parser.add_argument("--display-size", type=str, default="1280x720x24", help="Virtual display size WxHxD for Xvfb")
    parser.add_argument("--base-port", type=int, default=50039, help="Unity base port")
    parser.add_argument("--render-worker-id", type=int, default=0, help="Worker ID for render env")
    parser.add_argument("--team-a-worker-id", type=int, default=10, help="Unity worker ID for team A trainer env")
    parser.add_argument("--team-b-worker-id", type=int, default=20, help="Unity worker ID for team B trainer env")
    args = parser.parse_args()

    if args.team_a_strategy == "checkpoint" and not args.team_a_checkpoint:
        raise ValueError("team A strategy is checkpoint but no --team-a-checkpoint was provided.")
    if args.team_b_strategy == "checkpoint" and not args.team_b_checkpoint:
        raise ValueError("team B strategy is checkpoint but no --team-b-checkpoint was provided.")

    cfg = load_config(args.config)
    if not cfg:
        raise FileNotFoundError(f"Config not found or empty: {args.config}")

    if args.policy_mode is not None:
        cfg.setdefault("multiagent", {})["policy_mode"] = args.policy_mode

    base_env_config, ppo_template = _assemble_base_env_and_ppo_config(cfg, base_port=args.base_port)

    display_proc = _start_virtual_display_if_needed(
        headless=args.headless,
        display=args.display,
        size=args.display_size,
    )

    try:
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        tune.registry.register_env("Soccer", create_rllib_env)

        # Space probe: same env as training, unique worker to avoid collision.
        space_env = create_rllib_env(
            {**base_env_config, "worker_id": args.render_worker_id + 100}
        )
        space_env.close()

        # Render env (no fixed_unity_worker_id; plain dict has no RLlib worker_index).
        env = create_rllib_env(
            {**base_env_config, "render": True, "worker_id": args.render_worker_id}
        )

        team_trainers: Dict[str, Optional[PPOTrainer]] = {"A": None, "B": None}
        if args.team_a_strategy == "checkpoint":
            cfg_a = copy.deepcopy(ppo_template)
            cfg_a["env_config"] = {
                **base_env_config,
                "fixed_unity_worker_id": args.team_a_worker_id,
            }
            team_trainers["A"] = _build_trainer(args.team_a_checkpoint, cfg_a)
        if args.team_b_strategy == "checkpoint":
            cfg_b = copy.deepcopy(ppo_template)
            cfg_b["env_config"] = {
                **base_env_config,
                "fixed_unity_worker_id": args.team_b_worker_id,
            }
            team_trainers["B"] = _build_trainer(args.team_b_checkpoint, cfg_b)

        team_strategies = {"A": args.team_a_strategy, "B": args.team_b_strategy}
        team_policy_ids = {"A": args.team_a_policy_id, "B": args.team_b_policy_id}
        team_by_agent = {0: "A", 1: "A", 2: "B", 3: "B"}

        try:
            import imageio
        except ImportError as exc:
            raise ImportError("imageio is required for video output. Install with: pip install imageio") from exc

        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_trace)), exist_ok=True)

        obs = env.reset()
        start = time.time()
        traces = []
        with imageio.get_writer(args.output_video, fps=args.fps) as writer:
            while True:
                actions = {}
                for aid, aobs in obs.items():
                    aid_int = int(aid)
                    team = team_by_agent.get(aid_int, "A")
                    act = _get_action(
                        aid_int,
                        aobs,
                        team,
                        team_strategies=team_strategies,
                        team_trainers=team_trainers,
                        team_policy_ids=team_policy_ids,
                    )
                    if act is None:
                        act = env.action_space.sample()
                    actions[aid_int] = act

                obs, reward, done, info = env.step(actions)
                frame = None
                try:
                    frame = env.render(mode="rgb_array")
                except Exception:
                    frame = None

                if frame is not None:
                    writer.append_data(np.asarray(frame))

                traces.append(
                    {
                        "reward": {str(k): float(v) for k, v in reward.items()},
                        "done": done,
                    }
                )

                if _goal_scored(done, reward):
                    break
                if time.time() - start >= args.max_seconds:
                    break
                if isinstance(done, dict) and done.get("__all__", False):
                    break

        with open(args.output_trace, "w", encoding="utf-8") as f:
            json.dump(traces, f, indent=2)

        env.close()
        for trainer in team_trainers.values():
            if trainer is not None:
                trainer.stop()
        ray.shutdown()
        print(f"Saved video to: {os.path.abspath(args.output_video)}")
        print(f"Saved trace to: {os.path.abspath(args.output_trace)}")
    finally:
        if display_proc is not None:
            display_proc.terminate()
            try:
                display_proc.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    main()
