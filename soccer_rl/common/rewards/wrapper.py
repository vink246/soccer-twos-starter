from typing import Any, Dict, Optional

import gym

from .core import RewardComposer, RewardContext
from .terms import (
    BallGoalDistanceTerm,
    BallProgressTerm,
    OpponentPressureTerm,
    TrajectorySupportTerm,
)


TERM_REGISTRY = {
    "ball_goal_distance": BallGoalDistanceTerm,
    "ball_progress": BallProgressTerm,
    "trajectory_support": TrajectorySupportTerm,
    "opponent_pressure": OpponentPressureTerm,
}


def _to_agent_dict(value: Any, ref_obs: Dict[int, Any]) -> Dict[int, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {aid: value for aid in ref_obs.keys()}


def _normalize_info_per_agent(obs_dict: Dict[int, Any], info: Any) -> Dict[int, Any]:
    """
    soccer_twos returns per-agent info dicts for multiagent modes, but team_vs_policy
    returns a single flat dict with ball_info / player_info keys.
    """
    if not isinstance(info, dict):
        return {aid: {} for aid in obs_dict.keys()}
    aids = list(obs_dict.keys())
    if aids and all(aid in info for aid in aids):
        return {int(aid): info[aid] for aid in aids}
    if "ball_info" in info or "player_info" in info:
        return {aid: info for aid in aids}
    return {aid: info for aid in aids}


class RewardShapingWrapper(gym.Wrapper):
    """
    Env wrapper that composes sparse env reward with custom dense reward terms.
    """

    def __init__(self, env: gym.Env, reward_cfg: Dict[str, Any]):
        super().__init__(env)
        self.reward_cfg = reward_cfg or {}
        enabled_terms = self.reward_cfg.get("enabled_terms", [])
        if not enabled_terms:
            enabled_terms = ["ball_progress", "ball_goal_distance", "trajectory_support"]

        term_instances = []
        per_term_cfg = self.reward_cfg.get("terms", {})
        for term_name in enabled_terms:
            term_cls = TERM_REGISTRY.get(term_name)
            if term_cls is None:
                continue
            if per_term_cfg.get(term_name, {}).get("enabled", True):
                term_instances.append(term_cls())
        self.composer = RewardComposer(term_instances, self.reward_cfg)
        self._prev_obs: Optional[Dict[int, Any]] = None
        self._prev_info: Optional[Dict[int, Any]] = None
        self._step_idx = 0
        self._last_debug: Dict[str, Any] = {}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs_dict = obs if isinstance(obs, dict) else {0: obs}
        self._prev_obs = dict(obs_dict)
        self._prev_info = None
        self._step_idx = 0
        self.composer.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = obs if isinstance(obs, dict) else {0: obs}
        reward_dict = _to_agent_dict(reward, obs_dict)
        done_dict = _to_agent_dict(done, obs_dict)
        info_dict = _normalize_info_per_agent(obs_dict, info)

        ctx = RewardContext(
            prev_obs=self._prev_obs,
            prev_info=self._prev_info,
            obs=obs_dict,
            base_reward={aid: float(reward_dict.get(aid, 0.0)) for aid in obs_dict.keys()},
            info=info_dict,
            done=done_dict,
            step_idx=self._step_idx,
            cfg=self.reward_cfg,
        )
        result = self.composer.compose(ctx)
        shaped = result["per_agent"]

        # Store debugging payload for playback / diagnostics.
        self._last_debug = result

        self._prev_obs = dict(obs_dict)
        self._prev_info = dict(info_dict)
        self._step_idx += 1

        if isinstance(reward, dict):
            return obs, shaped, done, info
        return obs, shaped.get(0, float(reward)), done, info

    @property
    def last_reward_debug(self) -> Dict[str, Any]:
        return self._last_debug
