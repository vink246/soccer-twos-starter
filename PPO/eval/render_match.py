"""
Render Soccer-Twos PPO matches to video from checkpoints and/or random agents.

Examples:
  python PPO/eval/render_match.py --team-a-checkpoint <ckptA> --team-b-checkpoint <ckptB>
  python PPO/eval/render_match.py --team-a-checkpoint <ckptA> --team-b-strategy random
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training_utils import create_rllib_env  # noqa: E402


def _start_virtual_display_if_needed(headless: bool, display: str, size: str):
    if not headless:
        return None
    xvfb_path = shutil.which("Xvfb")
    if xvfb_path is None:
        raise RuntimeError(
            "Headless video requested but Xvfb was not found on PATH. "
            "Install Xvfb or run with xvfb-run."
        )
    # Do not overwrite an existing display if one already exists.
    os.environ.setdefault("DISPLAY", display)
    cmd = [xvfb_path, os.environ["DISPLAY"], "-screen", "0", size, "-nolisten", "tcp"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Give Xvfb a moment to initialize.
    time.sleep(0.5)
    return proc


def _policy_for_agent(agent_id: int, mode: str) -> str:
    if mode == "per_player":
        return f"player_{agent_id}"
    if mode == "team_shared":
        return "team_0" if agent_id in (0, 1) else "team_1"
    return "default"


def _goal_scored(done, reward, goal_threshold: float = 0.9) -> bool:
    if isinstance(done, dict) and done.get("__all__", False):
        if isinstance(reward, dict):
            return any(abs(float(v)) >= goal_threshold for v in reward.values())
        return abs(float(reward)) >= goal_threshold
    return False


def _build_trainer(
    checkpoint: str,
    policy_mode: str,
    env_config: Dict,
    obs_space,
    act_space,
):

    if policy_mode == "per_player":
        policies = {f"player_{i}": (None, obs_space, act_space, {}) for i in range(4)}
    elif policy_mode == "team_shared":
        policies = {
            "team_0": (None, obs_space, act_space, {}),
            "team_1": (None, obs_space, act_space, {}),
        }
    else:
        policies = {"default": (None, obs_space, act_space, {})}

    cfg = {
        "env": "Soccer",
        "framework": "torch",
        "num_workers": 0,
        "num_gpus": 0,
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda aid, *args, **kwargs: _policy_for_agent(
                int(aid), policy_mode
            ),
            "policies_to_train": [],
        },
    }
    trainer = PPOTrainer(config=cfg)
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
    parser.add_argument("--team-a-checkpoint", type=str, default=None)
    parser.add_argument("--team-b-checkpoint", type=str, default=None)
    parser.add_argument("--team-a-strategy", type=str, default="checkpoint", choices=["checkpoint", "random"])
    parser.add_argument("--team-b-strategy", type=str, default="checkpoint", choices=["checkpoint", "random"])
    parser.add_argument("--team-a-policy-id", type=str, default="team_0")
    parser.add_argument("--team-b-policy-id", type=str, default="team_1")
    parser.add_argument("--policy-mode", type=str, default="team_shared", choices=["team_shared", "per_player", "shared_all"])
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-video", type=str, default="PPO/runs/playback.mp4")
    parser.add_argument("--output-trace", type=str, default="PPO/runs/playback_trace.json")
    parser.add_argument("--headless", action="store_true", help="Use Xvfb and do not require an attached display")
    parser.add_argument("--display", type=str, default=":99", help="Display to use for headless mode")
    parser.add_argument("--display-size", type=str, default="1280x720x24", help="Virtual display size WxHxD for Xvfb")
    parser.add_argument("--base-port", type=int, default=50039, help="Unity base port")
    parser.add_argument("--render-worker-id", type=int, default=0, help="Worker ID for render env")
    parser.add_argument("--team-a-worker-id", type=int, default=10, help="Worker ID for team A trainer env")
    parser.add_argument("--team-b-worker-id", type=int, default=20, help="Worker ID for team B trainer env")
    args = parser.parse_args()

    if args.team_a_strategy == "checkpoint" and not args.team_a_checkpoint:
        raise ValueError("team A strategy is checkpoint but no --team-a-checkpoint was provided.")
    if args.team_b_strategy == "checkpoint" and not args.team_b_checkpoint:
        raise ValueError("team B strategy is checkpoint but no --team-b-checkpoint was provided.")

    display_proc = _start_virtual_display_if_needed(
        headless=args.headless,
        display=args.display,
        size=args.display_size,
    )

    try:
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        tune.registry.register_env("Soccer", create_rllib_env)

        env_config = {
            "variation": EnvType.multiagent_player,
            "multiagent": True,
            "flatten_branched": True,
            "base_port": args.base_port,
        }
        # Build spaces from a short-lived non-render env, then close it.
        space_env = create_rllib_env(
            {**env_config, "worker_id": args.render_worker_id + 100}
        )
        obs_space = space_env.observation_space
        act_space = space_env.action_space
        space_env.close()

        # Separate render env process/port from trainer env processes.
        env = create_rllib_env(
            {**env_config, "render": True, "worker_id": args.render_worker_id}
        )

        team_trainers = {"A": None, "B": None}
        if args.team_a_strategy == "checkpoint":
            team_trainers["A"] = _build_trainer(
                args.team_a_checkpoint,
                args.policy_mode,
                {**env_config, "fixed_unity_worker_id": args.team_a_worker_id},
                obs_space,
                act_space,
            )
        if args.team_b_strategy == "checkpoint":
            team_trainers["B"] = _build_trainer(
                args.team_b_checkpoint,
                args.policy_mode,
                {**env_config, "fixed_unity_worker_id": args.team_b_worker_id},
                obs_space,
                act_space,
            )

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
