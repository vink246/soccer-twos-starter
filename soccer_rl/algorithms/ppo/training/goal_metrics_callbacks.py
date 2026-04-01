from typing import Dict, Tuple

from ray.rllib.agents.callbacks import DefaultCallbacks


def _episode_env_goal_totals(episode) -> Tuple[float, float]:
    """Read cumulative env goal counts injected by EnvGoalTrackingWrapper (last step info)."""
    for aid in (0, 1, 2, 3, "0", "1", "2", "3", "agent0"):
        inf = episode.last_info_for(aid)
        if isinstance(inf, dict) and "env_goals_team_0" in inf:
            return float(inf["env_goals_team_0"]), float(inf["env_goals_team_1"])
    return 0.0, 0.0


class GoalStatsCallbacks(DefaultCallbacks):
    """
    Log true goals per episode from Unity sparse rewards (via EnvGoalTrackingWrapper).

    Custom metrics (per completed episode):
    - goals_team_0 / goals_team_1: env-reported cumulative goals that episode
    - goals_draw: 1.0 if both teams scored the same number (includes 0–0)
    """

    def on_episode_end(self, *, worker, base_env, policies: Dict, episode, **kwargs):
        g0, g1 = _episode_env_goal_totals(episode)
        draw = 1.0 if g0 == g1 else 0.0

        episode.custom_metrics["goals_team_0"] = g0
        episode.custom_metrics["goals_team_1"] = g1
        episode.custom_metrics["goals_draw"] = draw
