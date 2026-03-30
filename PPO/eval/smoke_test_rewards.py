"""
Run a short rollout and print reward-component breakdowns for debugging.

Examples:
  python PPO/eval/smoke_test_rewards.py --config PPO/configs/config.yaml
  python PPO/eval/smoke_test_rewards.py --steps 200 --print-every 10
"""

import argparse
import os
import sys
from typing import Any, Dict

import soccer_twos
from soccer_twos import EnvType

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PPO.rewards.wrapper import RewardShapingWrapper  # noqa: E402
from training_utils import load_config  # noqa: E402


def _as_dict(x: Any, obs: Dict[int, Any]) -> Dict[int, Any]:
    if isinstance(x, dict):
        return x
    return {aid: x for aid in obs.keys()}


def _env_variation(name: str):
    mapping = {
        "team_vs_policy": EnvType.team_vs_policy,
        "multiagent_team": EnvType.multiagent_team,
        "multiagent_player": EnvType.multiagent_player,
    }
    return mapping.get(name, EnvType.multiagent_player)


def main():
    parser = argparse.ArgumentParser(description="Reward shaping smoke test")
    parser.add_argument("--config", type=str, default="PPO/configs/config.yaml")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg.get("env") or {}
    reward_cfg = (env_cfg.get("reward") or {}).copy()
    reward_cfg["enabled"] = True

    base_env = soccer_twos.make(
        variation=_env_variation(env_cfg.get("variation", "multiagent_player")),
        multiagent=bool(env_cfg.get("multiagent", True)),
        single_player=bool(env_cfg.get("single_player", False)),
        flatten_branched=bool(env_cfg.get("flatten_branched", True)),
    )
    env = RewardShapingWrapper(base_env, reward_cfg)

    obs = env.reset()
    print(f"Running reward smoke test for {args.steps} steps")
    print(f"Enabled terms: {reward_cfg.get('enabled_terms', [])}")
    print("-" * 80)

    for step in range(args.steps):
        actions = {}
        for aid in obs.keys():
            actions[aid] = env.action_space.sample()

        obs, reward, done, _info = env.step(actions)
        reward_dict = _as_dict(reward, obs)
        dbg = env.last_reward_debug

        if step % args.print_every == 0:
            print(f"step={step}")
            print("  final_reward:", {k: round(float(v), 4) for k, v in reward_dict.items()})
            print(
                "  dense_reward:",
                {k: round(float(v), 4) for k, v in dbg.get("dense_per_agent", {}).items()},
            )
            per_term = dbg.get("per_term", {})
            for term_name, contrib in per_term.items():
                print(
                    f"  term[{term_name}]",
                    {k: round(float(v), 4) for k, v in contrib.items()},
                )
            print(
                "  team_rewards:",
                {k: round(float(v), 4) for k, v in dbg.get("team_rewards", {}).items()},
            )

        if isinstance(done, dict) and done.get("__all__", False):
            obs = env.reset()

    env.close()
    print("-" * 80)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
