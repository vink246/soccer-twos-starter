from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import RewardContext, RewardTerm


def _safe_array(obs: Any) -> Optional[np.ndarray]:
    if obs is None:
        return None
    try:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return arr
    except Exception:
        return None


def _xy_from_info(info_item: Any, key: str) -> Optional[np.ndarray]:
    if not isinstance(info_item, dict):
        return None
    value = info_item.get(key)
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 2:
            return arr[:2]
    except Exception:
        return None
    return None


def _xy_from_obs(obs: Any, indices: List[int]) -> Optional[np.ndarray]:
    arr = _safe_array(obs)
    if arr is None or len(indices) < 2:
        return None
    i0, i1 = int(indices[0]), int(indices[1])
    if i0 < 0 or i1 < 0 or i0 >= arr.shape[0] or i1 >= arr.shape[0]:
        return None
    return np.array([arr[i0], arr[i1]], dtype=np.float32)


def _norm_distance(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    return float(np.linalg.norm(a - b) / max(scale, 1e-6))


def _extract_positions(ctx: RewardContext) -> Dict[str, Any]:
    cfg = ctx.cfg.get("observation_indices", {})
    ball_idx = cfg.get("ball_xy", [0, 1])
    self_idx = cfg.get("self_xy", [2, 3])
    opp_goal_xy = np.asarray(ctx.cfg.get("opponent_goal_xy", [1.0, 0.0]), dtype=np.float32)

    out = {
        "ball_xy": {},
        "self_xy": {},
        "opp_goal_xy": opp_goal_xy,
    }
    for aid, obs in ctx.obs.items():
        info_item = ctx.info.get(aid, {}) if isinstance(ctx.info, dict) else {}
        ball_xy = _xy_from_info(info_item, "ball_position")
        self_xy = _xy_from_info(info_item, "player_position")
        if ball_xy is None:
            ball_xy = _xy_from_obs(obs, ball_idx)
        if self_xy is None:
            self_xy = _xy_from_obs(obs, self_idx)
        out["ball_xy"][aid] = ball_xy
        out["self_xy"][aid] = self_xy
    return out


class BallGoalDistanceTerm(RewardTerm):
    """
    Dense reward based on negative distance from ball to opponent goal.
    """

    name = "ball_goal_distance"

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        pos = _extract_positions(ctx)
        goal = pos["opp_goal_xy"]
        scale = float(ctx.cfg.get("distance_norm_scale", 1.5))
        rewards = {}
        for aid in ctx.obs.keys():
            ball_xy = pos["ball_xy"].get(aid)
            if ball_xy is None:
                rewards[aid] = 0.0
                continue
            dist = _norm_distance(ball_xy, goal, scale=scale)
            rewards[aid] = -dist
        return rewards


class BallProgressTerm(RewardTerm):
    """
    Potential-based reward: progress of ball towards opponent goal between steps.
    """

    name = "ball_progress"

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        if ctx.prev_obs is None:
            return {aid: 0.0 for aid in ctx.obs.keys()}
        cfg = ctx.cfg.get("observation_indices", {})
        ball_idx = cfg.get("ball_xy", [0, 1])
        goal = np.asarray(ctx.cfg.get("opponent_goal_xy", [1.0, 0.0]), dtype=np.float32)
        scale = float(ctx.cfg.get("distance_norm_scale", 1.5))
        rewards = {}
        for aid in ctx.obs.keys():
            prev_ball = _xy_from_obs(ctx.prev_obs.get(aid), ball_idx)
            cur_ball = _xy_from_obs(ctx.obs.get(aid), ball_idx)
            info_item = ctx.info.get(aid, {}) if isinstance(ctx.info, dict) else {}
            info_ball = _xy_from_info(info_item, "ball_position")
            if info_ball is not None:
                cur_ball = info_ball
            if prev_ball is None or cur_ball is None:
                rewards[aid] = 0.0
                continue
            prev_d = _norm_distance(prev_ball, goal, scale=scale)
            cur_d = _norm_distance(cur_ball, goal, scale=scale)
            rewards[aid] = prev_d - cur_d
        return rewards


class TrajectorySupportTerm(RewardTerm):
    """
    Reward players that stay close to the current ball->goal line.
    """

    name = "trajectory_support"

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        pos = _extract_positions(ctx)
        goal = pos["opp_goal_xy"]
        lane_width = float(ctx.cfg.get("trajectory_lane_width", 0.4))
        rewards = {}
        for aid in ctx.obs.keys():
            p = pos["self_xy"].get(aid)
            b = pos["ball_xy"].get(aid)
            if p is None or b is None:
                rewards[aid] = 0.0
                continue
            bg = goal - b
            norm = float(np.linalg.norm(bg))
            if norm < 1e-6:
                rewards[aid] = 0.0
                continue
            rel = p - b
            # Distance from point to line defined by b + t*bg.
            dist_line = abs(np.cross(bg, rel)) / norm
            rewards[aid] = max(0.0, 1.0 - dist_line / max(lane_width, 1e-6))
        return rewards


class OpponentPressureTerm(RewardTerm):
    """
    Approximate opponent mobility pressure:
    reward increases if nearest opponent is farther from the ball.
    """

    name = "opponent_pressure"

    def _default_opponents(self, aid: int) -> Tuple[int, int]:
        return (2, 3) if aid in (0, 1) else (0, 1)

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        pos = _extract_positions(ctx)
        scale = float(ctx.cfg.get("distance_norm_scale", 1.5))
        rewards = {}
        for aid in ctx.obs.keys():
            ball_xy = pos["ball_xy"].get(aid)
            if ball_xy is None:
                rewards[aid] = 0.0
                continue
            min_opp_dist = None
            for oid in self._default_opponents(int(aid)):
                opp_xy = pos["self_xy"].get(oid)
                if opp_xy is None:
                    continue
                d = _norm_distance(opp_xy, ball_xy, scale=scale)
                if min_opp_dist is None or d < min_opp_dist:
                    min_opp_dist = d
            if min_opp_dist is None:
                rewards[aid] = 0.0
            else:
                rewards[aid] = min_opp_dist
        return rewards
