"""
Wrap Soccer-Twos env to expose true goal counts from Unity sparse rewards.

Unity signals goals with large positive/negative rewards (typically ±1 per agent).
We increment counters when team-aggregated or per-player rewards cross
``goal_reward_threshold`` (same default as reward shaping: 0.9).
"""

from typing import Any, Dict, Tuple, Union

import gym


RewardDict = Dict[Any, float]


class EnvGoalTrackingWrapper(gym.Wrapper):
    """Accumulate goals per team from raw env rewards; attach to each step's info."""

    def __init__(self, env: gym.Env, goal_reward_threshold: float = 0.9):
        super().__init__(env)
        self._th = float(goal_reward_threshold)
        self._goals_team_0 = 0
        self._goals_team_1 = 0

    def reset(self, **kwargs):
        self._goals_team_0 = 0
        self._goals_team_1 = 0
        return self.env.reset(**kwargs)

    def _step_deltas(self, reward: Union[float, RewardDict]) -> Tuple[int, int]:
        """Return (delta_team_0_goals, delta_team_1_goals) for this timestep."""
        th = self._th
        if isinstance(reward, dict):
            r: Dict[int, float] = {}
            for k, v in reward.items():
                try:
                    ik = int(k)
                except (TypeError, ValueError):
                    continue
                r[ik] = float(v)

            if {0, 1, 2, 3}.issubset(r.keys()):
                t0 = max(r[0], r[1])
                t1 = max(r[2], r[3])
                d0 = int(t0 >= th)
                d1 = int(t1 >= th)
                return d0, d1
            if r.keys() == {0, 1}:
                d0 = int(r[0] >= th)
                d1 = int(r[1] >= th)
                return d0, d1
            return 0, 0

        s = float(reward)
        # team_vs_policy: scalar is controlled-team sum; sign indicates score / concede
        if s >= th:
            return 1, 0
        if s <= -th:
            return 0, 1
        return 0, 0

    def _payload(self) -> Dict[str, float]:
        return {
            "env_goals_team_0": float(self._goals_team_0),
            "env_goals_team_1": float(self._goals_team_1),
        }

    def _merge_info(self, obs, info, payload: Dict[str, float]):
        if isinstance(obs, dict):
            merged = {}
            for aid in obs.keys():
                base = {}
                if isinstance(info, dict) and aid in info and isinstance(info[aid], dict):
                    base = dict(info[aid])
                merged[aid] = {**base, **payload}
            return merged
        base = dict(info) if isinstance(info, dict) else {}
        return {**base, **payload}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        d0, d1 = self._step_deltas(reward)
        self._goals_team_0 += d0
        self._goals_team_1 += d1
        payload = self._payload()
        new_info = self._merge_info(obs, info, payload)
        return obs, reward, done, new_info
