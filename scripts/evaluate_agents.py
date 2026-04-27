"""
Run head-to-head evaluation between two AgentInterface packages.

Example:
  conda run -n soccertwos python scripts/evaluate_agents.py \
    --agent1 agents.ppo_dense_agent_ceia_trained \
    --agent2 agents.ceia_baseline_agent \
    --games 20
"""

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import soccer_twos
from soccer_twos import AgentInterface, EnvType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_agent_class(module_path: str):
    mod = importlib.import_module(module_path)
    candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj is AgentInterface:
            continue
        if issubclass(obj, AgentInterface):
            candidates.append(obj)
    if not candidates:
        raise ValueError(f"No AgentInterface subclass found in module {module_path!r}")
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise ValueError(
            f"Multiple AgentInterface classes found in {module_path!r}: {names}. "
            "Please expose only one in __init__.py"
        )
    return candidates[0]


def _extract_two_actions(actions: Dict[Any, Any], who: str) -> Tuple[Any, Any]:
    if not isinstance(actions, dict):
        raise TypeError(f"{who}.act(...) must return dict, got {type(actions)}")
    keys = sorted(actions.keys())
    if len(keys) != 2:
        raise ValueError(f"{who}.act(...) must return exactly 2 actions, got keys={keys}")
    return actions[keys[0]], actions[keys[1]]


def run_games(agent1_module: str, agent2_module: str, games: int, render: bool) -> None:
    env = soccer_twos.make(
        variation=EnvType.multiagent_player,
        # Match soccer_twos.watch-style AgentInterface behavior (branched actions).
        flatten_branched=False,
        render=render,
    )
    try:
        agent1_cls = _load_agent_class(agent1_module)
        agent2_cls = _load_agent_class(agent2_module)
        agent1 = agent1_cls(env)
        agent2 = agent2_cls(env)

        wins_1 = 0
        wins_2 = 0
        ties = 0
        per_game_rows: List[Tuple[int, int, int, float, float, str]] = []

        for game_idx in range(1, games + 1):
            obs = env.reset()
            done_all = False
            team_returns = [0.0, 0.0]
            goals = [0, 0]

            while not done_all:
                obs_1 = {0: obs[0], 1: obs[1]}
                obs_2 = {0: obs[2], 1: obs[3]}

                a1_raw = agent1.act(obs_1)
                a2_raw = agent2.act(obs_2)
                a10, a11 = _extract_two_actions(a1_raw, "agent1")
                a20, a21 = _extract_two_actions(a2_raw, "agent2")

                obs, rew, done, _info = env.step({0: a10, 1: a11, 2: a20, 3: a21})
                r1 = float(rew[0]) + float(rew[1])
                r2 = float(rew[2]) + float(rew[3])
                team_returns[0] += r1
                team_returns[1] += r2

                # Sparse Soccer-Twos goals usually appear as opposite-sign team rewards.
                if r1 > 0.0 and r2 < 0.0:
                    goals[0] += 1
                elif r2 > 0.0 and r1 < 0.0:
                    goals[1] += 1

                done_all = bool(done.get("__all__", False)) if isinstance(done, dict) else bool(done)

            if goals[0] > goals[1]:
                wins_1 += 1
                winner = "agent1"
            elif goals[1] > goals[0]:
                wins_2 += 1
                winner = "agent2"
            else:
                # If inferred goals tie, use team return tie-break (still report tie if equal).
                if team_returns[0] > team_returns[1]:
                    wins_1 += 1
                    winner = "agent1 (tie-break: return)"
                elif team_returns[1] > team_returns[0]:
                    wins_2 += 1
                    winner = "agent2 (tie-break: return)"
                else:
                    ties += 1
                    winner = "tie"

            per_game_rows.append(
                (game_idx, goals[0], goals[1], team_returns[0], team_returns[1], winner)
            )

        print(f"\nMatchup: {agent1_module} (team0) vs {agent2_module} (team1)")
        print(f"Games: {games}\n")
        print("Per-game results:")
        for row in per_game_rows:
            idx, g1, g2, ret1, ret2, winner = row
            print(
                f"  Game {idx:02d}: score {g1}-{g2} | returns {ret1:.3f} vs {ret2:.3f} | winner: {winner}"
            )

        print("\nSummary:")
        print(f"  {agent1_module} wins: {wins_1}")
        print(f"  {agent2_module} wins: {wins_2}")
        print(f"  ties: {ties}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate two soccer_twos AgentInterface modules over multiple games."
    )
    parser.add_argument("--agent1", required=True, help="Team 0 agent module path")
    parser.add_argument("--agent2", required=True, help="Team 1 agent module path")
    parser.add_argument("--games", type=int, default=20, help="Number of games (default: 20)")
    parser.add_argument("--render", action="store_true", help="Render evaluation rollout")
    args = parser.parse_args()

    if args.games < 1:
        raise ValueError("--games must be >= 1")

    run_games(args.agent1, args.agent2, args.games, args.render)


if __name__ == "__main__":
    main()
