"""Gym wrapper that adds YAML-configurable dense rewards on top of Unity sparse rewards."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from soccer_rl.dense_rewards import (
    WorldSnapshot,
    compute_dense_total,
    parse_dense_reward_config,
)


def _pick_obs_info_for_agent(
    agent_key: Any,
    obs_agent: np.ndarray,
    info_branch: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    o = np.asarray(obs_agent, dtype=np.float64).reshape(-1)
    if not isinstance(info_branch, dict):
        return o, {}
    if "player_info" in info_branch:
        return o, info_branch
    if 0 in info_branch and isinstance(info_branch[0], dict) and "player_info" in info_branch[0]:
        half = o.size // 2
        if half > 0:
            return o[:half], info_branch[0]
        return o, info_branch[0]
    return o, info_branch


class DenseRewardWrapper(gym.Wrapper):
    """
    ``reward = sparse_weight * env_reward + sum_i weight_i * term_i``.

    Works with scalar rewards (``team_vs_policy`` + ``single_player``) and with
    dict rewards (``multiagent_player``, ``multiagent_team``). Per-agent
    ``prev`` state is tracked separately for delta-based terms.
    """

    def __init__(self, env: gym.Env, dense_cfg: Optional[Dict[str, Any]]):
        super().__init__(env)
        self._cfg = parse_dense_reward_config(dense_cfg or {})
        self._prev: Dict[Any, Optional[WorldSnapshot]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.get("enabled"))

    def reset(self, **kwargs):
        self._prev.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self._cfg.get("enabled"):
            return obs, reward, done, info

        sparse_w = float(self._cfg["sparse_weight"])
        terms = self._cfg["terms"]
        clip = self._cfg.get("clip")
        clip_f = float(clip) if clip is not None else None

        if isinstance(reward, dict):
            new_reward = {}
            for k in reward:
                o_k = obs[k]
                inf_k = info.get(k, {}) if isinstance(info, dict) else {}
                o_use, i_use = _pick_obs_info_for_agent(k, o_k, inf_k)
                prev = self._prev.get(k)
                dense, snap = compute_dense_total(o_use, i_use, prev, terms)
                self._prev[k] = snap
                if clip_f is not None:
                    dense = float(np.clip(dense, -clip_f, clip_f))
                new_reward[k] = float(reward[k]) * sparse_w + dense
            return obs, new_reward, done, info

        o_vec = np.asarray(obs, dtype=np.float64).reshape(-1)
        inf_d = info if isinstance(info, dict) else {}
        prev = self._prev.get(None)
        dense, snap = compute_dense_total(o_vec, inf_d, prev, terms)
        self._prev[None] = snap
        if clip_f is not None:
            dense = float(np.clip(dense, -clip_f, clip_f))
        return obs, float(reward) * sparse_w + dense, done, info


def maybe_wrap_dense_reward(env: gym.Env, full_config: Dict[str, Any]) -> gym.Env:
    dr = full_config.get("dense_reward")
    if not dr or not dr.get("enabled"):
        return env
    return DenseRewardWrapper(env, dr)
