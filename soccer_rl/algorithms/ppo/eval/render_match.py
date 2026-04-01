"""
Render Soccer-Twos PPO matches to video from checkpoints and/or random agents.

Restore requires the same RLlib config as training (model, batch sizes, multiagent).
Pass --config pointing to the same YAML you used for training.

Examples:
  conda activate soccertwos  # env must have soccer_twos + Unity binaries
  python soccer_rl/algorithms/ppo/eval/render_match.py \\
    --config soccer_rl/algorithms/ppo/configs/config.yaml \\
    --team-a-checkpoint <ckptA> --team-b-strategy random --headless

Video uses the watch-soccer-twos Unity build (camera). The default training build has no
visual observations, so render(rgb_array) would otherwise stay None.
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from soccer_rl.algorithms.ppo.ppo_inference_config import build_ppo_restore_configs  # noqa: E402
from soccer_rl.common.playback import start_virtual_display_if_needed  # noqa: E402
from soccer_rl.common.training_utils import create_rllib_env, load_config  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "..", "..", "configs", "config.yaml")


def _goal_scored(done, reward, goal_threshold: float = 0.9) -> bool:
    if isinstance(done, dict) and done.get("__all__", False):
        if isinstance(reward, dict):
            return any(abs(float(v)) >= goal_threshold for v in reward.values())
        return abs(float(reward)) >= goal_threshold
    return False


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
    action = trainer.compute_action(obs, policy_id=policy_id)
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
    parser.add_argument("--output-video", type=str, default="runs/playback.mp4")
    parser.add_argument("--output-trace", type=str, default="runs/playback_trace.json")
    parser.add_argument(
        "--frame-log-interval",
        type=int,
        default=1,
        metavar="N",
        help="Print 'Rendered frame k' every N frames actually written (1 = every frame; use 0 to disable)",
    )
    parser.add_argument("--headless", action="store_true", help="Use Xvfb and do not require an attached display")
    parser.add_argument("--display", type=str, default=":99", help="Display to use for headless mode")
    parser.add_argument("--display-size", type=str, default="1280x720x24", help="Virtual display size WxHxD for Xvfb")
    parser.add_argument("--base-port", type=int, default=50039, help="Unity base port")
    parser.add_argument("--render-worker-id", type=int, default=0, help="Worker ID for render env")
    parser.add_argument("--team-a-worker-id", type=int, default=10, help="Unity worker ID for team A trainer env")
    parser.add_argument("--team-b-worker-id", type=int, default=20, help="Unity worker ID for team B trainer env")
    parser.add_argument(
        "--no-watch-env",
        action="store_true",
        help="Use training Unity binary for the render env (no camera; rgb_array will be empty). "
        "Default is watch-soccer-twos so MP4 capture works.",
    )
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

    base_env_config, ppo_template = build_ppo_restore_configs(
        cfg, args.base_port, force_single_env_per_worker=True
    )

    display_proc = start_virtual_display_if_needed(
        headless=args.headless,
        display=args.display,
        size=args.display_size,
    )

    try:
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        tune.registry.register_env("Soccer", create_rllib_env)

        space_env = create_rllib_env(
            {**base_env_config, "worker_id": args.render_worker_id + 100}
        )
        space_env.close()

        team_trainers: Dict[str, Optional[PPOTrainer]] = {"A": None, "B": None}
        if args.team_a_strategy == "checkpoint" or args.team_b_strategy == "checkpoint":
            print(
                "[render_match] Loading checkpoint(s) and headless Unity for policies "
                "(often 1–3 minutes; no window yet)…",
                flush=True,
            )
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

        render_env_cfg: Dict[str, Any] = {
            **base_env_config,
            "render": True,
            "worker_id": args.render_worker_id,
        }
        if not args.no_watch_env:
            render_env_cfg["watch"] = True
            print(
                "[render_match] Using watch-soccer-twos (watch=True) for camera frames.",
                flush=True,
            )
        print(
            "[render_match] Starting video Unity window. If it shows PAUSED, wait for "
            "the first step; if it never moves, click inside the window and try Space or P "
            "(some builds need focus to run).",
            flush=True,
        )
        env = create_rllib_env(render_env_cfg)

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
        frames_written = 0
        render_fail_logged = False
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
                except Exception as exc:
                    if not render_fail_logged:
                        print(
                            f"[render_match] env.render(rgb_array) failed once: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        render_fail_logged = True
                    frame = None

                if frame is None and not render_fail_logged:
                    print(
                        "[render_match] env.render returned None (no rgb_array); video may be empty.",
                        flush=True,
                    )
                    render_fail_logged = True

                if frame is not None:
                    writer.append_data(np.asarray(frame))
                    frames_written += 1
                    interval = args.frame_log_interval
                    if interval > 0 and frames_written % interval == 0:
                        print(f"Rendered frame {frames_written}", flush=True)

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
        print(
            f"Video frames written: {frames_written} (trace steps: {len(traces)})",
            flush=True,
        )
        if frames_written == 0:
            print(
                "WARNING: No frames were captured; the MP4 may be empty or invalid. "
                "If you passed --no-watch-env, remove it (training build has no camera). "
                "Otherwise check headless/Xvfb and DISPLAY.",
                flush=True,
            )
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
