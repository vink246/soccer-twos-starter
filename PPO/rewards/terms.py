from typing import Any, Dict, List, Optional

import numpy as np

from .core import RewardContext, RewardTerm


def _xy_from_player_ball_info(info_item: Any, key: str) -> Optional[np.ndarray]:
    if not isinstance(info_item, dict):
        return None
    nested = info_item.get(key)
    if isinstance(nested, dict):
        val = nested.get("position")
        if val is not None:
            try:
                arr = np.asarray(val, dtype=np.float32).reshape(-1)
                if arr.shape[0] >= 2:
                    return arr[:2]
            except Exception:
                return None
    return None


def _norm_distance(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> float:
    return float(np.linalg.norm(a - b) / max(scale, 1e-6))


def _extract_positions(ctx: RewardContext) -> Dict[str, Any]:
    opp_goal_xy = np.asarray(ctx.cfg.get("opponent_goal_xy", [1.0, 0.0]), dtype=np.float32)

    out = {
        "ball_xy": {},
        "self_xy": {},
        "opp_goal_xy": opp_goal_xy,
    }
    for aid in ctx.obs.keys():
        info_item = ctx.info.get(aid, {}) if isinstance(ctx.info, dict) else {}
        ball_xy = _xy_from_player_ball_info(info_item, "ball_info")
        self_xy = _xy_from_player_ball_info(info_item, "player_info")

        # multiagent_team: info[team_id][player_local_id] -> per-player dicts
        if (ball_xy is None or self_xy is None) and isinstance(info_item, dict):
            nested_players = [v for v in info_item.values() if isinstance(v, dict)]
            if nested_players:
                player_positions = []
                for player_entry in nested_players:
                    bxy = _xy_from_player_ball_info(player_entry, "ball_info")
                    pxy = _xy_from_player_ball_info(player_entry, "player_info")
                    if ball_xy is None and bxy is not None:
                        ball_xy = bxy
                    if pxy is not None:
                        player_positions.append(pxy)
                if self_xy is None and player_positions:
                    self_xy = np.mean(np.stack(player_positions, axis=0), axis=0)

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
    Uses info.ball_info.position only (no observation-vector indices).
    """

    name = "ball_progress"

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        if ctx.prev_info is None:
            return {aid: 0.0 for aid in ctx.obs.keys()}
        goal = np.asarray(ctx.cfg.get("opponent_goal_xy", [1.0, 0.0]), dtype=np.float32)
        scale = float(ctx.cfg.get("distance_norm_scale", 1.5))
        rewards = {}
        for aid in ctx.obs.keys():
            prev_ball = None
            if isinstance(ctx.prev_info, dict):
                prev_ball = _xy_from_player_ball_info(ctx.prev_info.get(aid, {}), "ball_info")
            cur_ball = _xy_from_player_ball_info(
                ctx.info.get(aid, {}) if isinstance(ctx.info, dict) else {},
                "ball_info",
            )
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
            dist_line = abs(np.cross(bg, rel)) / norm
            rewards[aid] = max(0.0, 1.0 - dist_line / max(lane_width, 1e-6))
        return rewards


class OpponentPressureTerm(RewardTerm):
    """
    Approximate opponent mobility pressure:
    reward increases if nearest opponent is farther from the ball.
    """

    name = "opponent_pressure"

    def _opponent_ids(self, aid: int, all_ids: List[int], team_map: Dict[str, List[int]]) -> List[int]:
        if team_map:
            own_team = None
            for _, members in team_map.items():
                if aid in members:
                    own_team = members
                    break
            if own_team is not None:
                return [x for x in all_ids if x not in own_team]
        if len(all_ids) == 2:
            return [x for x in all_ids if x != aid]
        if len(all_ids) == 4:
            return [2, 3] if aid in (0, 1) else [0, 1]
        return [x for x in all_ids if x != aid]

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        pos = _extract_positions(ctx)
        scale = float(ctx.cfg.get("distance_norm_scale", 1.5))
        all_ids = sorted(int(a) for a in ctx.obs.keys())
        team_map_cfg = ctx.cfg.get("team_map", {})
        team_map = {}
        if isinstance(team_map_cfg, dict):
            for key, val in team_map_cfg.items():
                if isinstance(val, list):
                    team_map[key] = [int(x) for x in val]
        rewards = {}
        for aid in ctx.obs.keys():
            ball_xy = pos["ball_xy"].get(aid)
            if ball_xy is None:
                rewards[aid] = 0.0
                continue
            min_opp_dist = None
            for oid in self._opponent_ids(int(aid), all_ids, team_map):
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
