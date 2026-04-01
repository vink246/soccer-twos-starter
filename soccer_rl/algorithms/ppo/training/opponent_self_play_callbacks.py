"""
Periodically set team_vs_policy opponent to the current learned policy (self-play).

Uses ``result["episodes_total"]`` (cumulative completed episodes across workers) and
``env.team_vs_self_play.swap_every_n_episodes``: after each multiple of N, all rollout
workers call ``TeamVsPolicyWrapper.set_opponent_policy`` with ``policy.compute_single_action``.

Requires ``env.single_player: true`` so opponent observations (336-dim) match the policy
input (the learned policy acts for one player; the same weights are used for both
opponent players).
"""

from typing import Any, Dict, Optional

from ray.rllib.agents.callbacks import DefaultCallbacks

try:
    from soccer_twos.wrappers import TeamVsPolicyWrapper
except Exception:  # pragma: no cover
    TeamVsPolicyWrapper = None


def find_team_vs_policy_wrapper(env) -> Optional[Any]:
    if TeamVsPolicyWrapper is None:
        return None
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, TeamVsPolicyWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _self_play_cfg(trainer) -> Dict[str, Any]:
    cfg = trainer.config.get("env_config") or {}
    return cfg.get("team_vs_self_play") or {}


def sync_opponent_policy_to_learned(trainer, policy_id: str = "default", explore: bool = False) -> None:
    """On every rollout worker, point TeamVsPolicy opponent at the local policy."""

    def update_worker(worker):
        policy = worker.get_policy(policy_id)
        if policy is None:
            return [False]

        def per_sub_env(env):
            tv = find_team_vs_policy_wrapper(env)
            if tv is None:
                return False

            def opponent_policy(obs):
                return policy.compute_single_action(obs, explore=explore)[0]

            tv.set_opponent_policy(opponent_policy)
            return True

        return worker.foreach_env(per_sub_env)

    trainer.workers.foreach_worker(update_worker)


class OpponentSelfPlayCallbacks(DefaultCallbacks):
    """Swap opponent from random → current policy every N completed training episodes."""

    def __init__(self, legacy_callbacks_dict=None):
        super().__init__(legacy_callbacks_dict)
        self._last_episode_bucket = -1

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        sp = _self_play_cfg(trainer)
        if not sp.get("enabled", False):
            return

        n = int(sp.get("swap_every_n_episodes", 0) or 0)
        if n <= 0:
            return

        ep_total = int(result.get("episodes_total") or 0)
        bucket = ep_total // n
        if bucket <= self._last_episode_bucket:
            return
        self._last_episode_bucket = bucket
        if bucket < 1:
            return

        policy_id = str(sp.get("policy_id", "default"))
        explore = bool(sp.get("opponent_stochastic", False))
        sync_opponent_policy_to_learned(trainer, policy_id=policy_id, explore=explore)
        result["opponent_self_play_sync_count"] = bucket
