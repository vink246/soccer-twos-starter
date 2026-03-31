from typing import Dict

from ray.rllib.agents.callbacks import DefaultCallbacks


class GoalStatsCallbacks(DefaultCallbacks):
    """
    Track goals scored vs conceded per episode via custom_metrics.

    We approximate:
    - goals_for = 1 when episode return > +0.5
    - goals_against = 1 when episode return < -0.5
    - draws = 1 when neither threshold is crossed
    """

    def on_episode_end(self, *, worker, base_env, policies: Dict, episode, **kwargs):
        try:
            total_return = float(episode.get_return())
        except Exception:
            total_return = 0.0

        goals_for = 1.0 if total_return > 0.5 else 0.0
        goals_against = 1.0 if total_return < -0.5 else 0.0
        draw = 1.0 if goals_for == 0.0 and goals_against == 0.0 else 0.0

        episode.custom_metrics["goals_for"] = goals_for
        episode.custom_metrics["goals_against"] = goals_against
        episode.custom_metrics["goals_draw"] = draw

