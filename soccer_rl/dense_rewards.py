"""
Configurable dense reward terms for Soccer-Twos training.

Uses ``player_info`` / ``ball_info`` from ``info`` when the Unity binary appends
the extra 9 floats (345-dim raw vector). If only 336-dim observations are
available, geometric terms return 0.

Optional flat-observation indices let you wire opponent-relative features
(e.g. approximate line-of-sight) once you know the layout for your build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

TermFn = Callable[["WorldSnapshot", Optional["WorldSnapshot"], np.ndarray, Mapping[str, Any]], float]


@dataclass
class WorldSnapshot:
    player_xy: Optional[np.ndarray]
    ball_xy: Optional[np.ndarray]
    ball_vxy: Optional[np.ndarray]
    player_yaw: Optional[float]


def _as_vec2(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return None
    return a[:2].copy()


def snapshot_from_info(info: Mapping[str, Any]) -> WorldSnapshot:
    if not isinstance(info, Mapping):
        return WorldSnapshot(None, None, None, None)
    pi = info.get("player_info") or {}
    bi = info.get("ball_info") or {}
    pos_p = _as_vec2(pi.get("position"))
    pos_b = _as_vec2(bi.get("position"))
    vel_b = _as_vec2(bi.get("velocity"))
    yaw = pi.get("rotation_y")
    try:
        yaw_f = float(yaw) if yaw is not None else None
    except (TypeError, ValueError):
        yaw_f = None
    return WorldSnapshot(pos_p, pos_b, vel_b, yaw_f)


def _attack_unit(cfg: Mapping[str, Any]) -> np.ndarray:
    axis = int(cfg.get("axis", 0))
    sign = float(cfg.get("attack_sign", 1.0))
    u = np.zeros(2, dtype=np.float64)
    u[axis % 2] = sign
    return u


def _own_goal_xy(cfg: Mapping[str, Any]) -> np.ndarray:
    g = cfg.get("own_goal_xy", [-1.0, 0.0])
    a = np.asarray(g, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return np.array([-1.0, 0.0], dtype=np.float64)
    return a[:2].copy()


def _opp_goal_xy(cfg: Mapping[str, Any]) -> np.ndarray:
    g = cfg.get("opponent_goal_xy", [1.0, 0.0])
    a = np.asarray(g, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    return a[:2].copy()


def term_ball_attack_axis_delta(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    del obs
    if s.ball_xy is None or prev is None or prev.ball_xy is None:
        return 0.0
    u = _attack_unit(cfg)
    return float(np.dot(s.ball_xy - prev.ball_xy, u))


def term_ball_vel_attack_component(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    del prev, obs
    if s.ball_vxy is None:
        return 0.0
    u = _attack_unit(cfg)
    return float(np.dot(s.ball_vxy, u))


def term_ball_own_goal_threat_gaussian(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    del prev, obs
    if s.ball_xy is None:
        return 0.0
    og = _own_goal_xy(cfg)
    sigma = float(cfg.get("sigma", 0.35))
    sigma = max(sigma, 1e-6)
    d = float(np.linalg.norm(s.ball_xy - og))
    return float(-math.exp(-0.5 * (d / sigma) ** 2))


def term_distance_to_ball_closer_delta(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    del obs
    if s.player_xy is None or s.ball_xy is None:
        return 0.0
    dist = float(np.linalg.norm(s.player_xy - s.ball_xy))
    if prev is None or prev.player_xy is None or prev.ball_xy is None:
        return 0.0
    prev_dist = float(np.linalg.norm(prev.player_xy - prev.ball_xy))
    scale = float(cfg.get("distance_scale", 1.0))
    scale = max(scale, 1e-6)
    return (prev_dist - dist) / scale


def term_screen_own_goal(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    """Smooth bonus when the agent sits between the ball and own goal on the attack axis."""
    del prev, obs
    if s.player_xy is None or s.ball_xy is None:
        return 0.0
    u = _attack_unit(cfg)
    og = _own_goal_xy(cfg)
    axis = int(cfg.get("axis", 0))
    # Project positions on attack axis (1D ordering toward opponent goal)
    g = float(np.dot(og, u))
    b = float(np.dot(s.ball_xy, u))
    p = float(np.dot(s.player_xy, u))
    lo, hi = (g, b) if g <= b else (b, g)
    margin = min(p - lo, hi - p)
    k = float(cfg.get("sigmoid_k", 8.0))
    if margin >= 0:
        return float(1.0 / (1.0 + math.exp(-k * margin)))
    return float(1.0 / (1.0 + math.exp(k * margin)) - 1.0)


def term_ball_opponent_goal_potential_delta(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    """Change in negative distance from ball to opponent goal (encourages moving ball toward goal)."""
    del obs
    if s.ball_xy is None or prev is None or prev.ball_xy is None:
        return 0.0
    og = _opp_goal_xy(cfg)
    d_now = float(np.linalg.norm(s.ball_xy - og))
    d_prev = float(np.linalg.norm(prev.ball_xy - og))
    return d_prev - d_now


def term_ball_distance_to_own_goal(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    del prev, obs
    if s.ball_xy is None:
        return 0.0
    return float(np.linalg.norm(s.ball_xy - _own_goal_xy(cfg)))


def term_ball_distance_to_opponent_goal(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    del prev, obs
    if s.ball_xy is None:
        return 0.0
    return float(np.linalg.norm(s.ball_xy - _opp_goal_xy(cfg)))


def term_ball_own_times_opp_goal_distance(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    """
    ``dist(ball, own_goal) * dist(ball, opponent_goal)`` — use a *negative* weight in YAML to
    encourage keeping the ball away from your goal while it is still far from the opponent goal
    (explore composite shaping; scale with small ``weight``).
    """
    del prev, obs
    if s.ball_xy is None:
        return 0.0
    d0 = float(np.linalg.norm(s.ball_xy - _own_goal_xy(cfg)))
    d1 = float(np.linalg.norm(s.ball_xy - _opp_goal_xy(cfg)))
    return d0 * d1


def _obs_xy_yaw(
    obs: np.ndarray, cfg: Mapping[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    px = cfg.get("opponent_pos_obs")
    yaw_i = cfg.get("opponent_yaw_obs")
    if px is None or yaw_i is None:
        return None, None
    idx = list(px)
    if len(idx) != 2:
        return None, None
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    try:
        ix, iy = int(idx[0]), int(idx[1])
        iyaw = int(yaw_i)
        if ix < 0 or iy < 0 or iyaw < 0 or ix >= o.size or iy >= o.size or iyaw >= o.size:
            return None, None
        pos = np.array([o[ix], o[iy]], dtype=np.float64)
        yaw = float(o[iyaw])
        return pos, yaw
    except (TypeError, ValueError, IndexError):
        return None, None


def _yaw_to_forward(yaw: float, cfg: Mapping[str, Any]) -> np.ndarray:
    if cfg.get("yaw_in_degrees", True):
        yaw = math.radians(yaw)
    # Tunable: swap if your Unity build uses a different heading convention
    if cfg.get("forward_is_sin_cos", True):
        return np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def term_hide_ball_from_opponent_los(
    s: WorldSnapshot, prev: Optional[WorldSnapshot], obs: np.ndarray, cfg: Mapping[str, Any]
) -> float:
    """
    Positive when the opponent likely does *not* have the ball in front of them.

    Returns (1 - visibility) in [0, 1] if opponent position & yaw are read from ``obs``;
    otherwise 0. Weight in YAML scales this value.
    """
    del prev
    if s.ball_xy is None:
        return 0.0
    opp_pos, opp_yaw = _obs_xy_yaw(obs, cfg)
    if opp_pos is None or opp_yaw is None:
        return 0.0
    to_ball = s.ball_xy - opp_pos
    dist = float(np.linalg.norm(to_ball))
    max_range = float(cfg.get("max_range", 100.0))
    if dist < 1e-8 or dist > max_range:
        return 1.0
    to_ball_u = to_ball / dist
    fwd = _yaw_to_forward(opp_yaw, cfg)
    cos_half = math.cos(math.radians(float(cfg.get("fov_degrees", 120.0)) * 0.5))
    visible = float(np.dot(to_ball_u, fwd)) >= cos_half
    return 1.0 - (1.0 if visible else 0.0)


TERM_REGISTRY: Dict[str, TermFn] = {
    "ball_attack_axis_delta": term_ball_attack_axis_delta,
    "ball_vel_attack_component": term_ball_vel_attack_component,
    "ball_own_goal_threat_gaussian": term_ball_own_goal_threat_gaussian,
    "distance_to_ball_closer_delta": term_distance_to_ball_closer_delta,
    "screen_own_goal": term_screen_own_goal,
    "ball_opponent_goal_potential_delta": term_ball_opponent_goal_potential_delta,
    "ball_distance_to_own_goal": term_ball_distance_to_own_goal,
    "ball_distance_to_opponent_goal": term_ball_distance_to_opponent_goal,
    "ball_own_times_opp_goal_distance": term_ball_own_times_opp_goal_distance,
    "hide_ball_from_opponent_los": term_hide_ball_from_opponent_los,
}


def _normalize_terms_cfg(raw: Any) -> List[Tuple[str, Dict[str, Any], float]]:
    out: List[Tuple[str, Dict[str, Any], float]] = []
    if raw is None:
        return out
    if isinstance(raw, Mapping):
        for name, spec in raw.items():
            if not isinstance(spec, Mapping):
                continue
            w = float(spec.get("weight", 0.0))
            if w == 0.0:
                continue
            term_cfg = {k: v for k, v in spec.items() if k != "weight"}
            out.append((str(name), term_cfg, w))
        return out
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping) or "name" not in item:
                continue
            name = str(item["name"])
            w = float(item.get("weight", 0.0))
            if w == 0.0:
                continue
            term_cfg = {k: v for k, v in item.items() if k not in ("name", "weight")}
            out.append((name, term_cfg, w))
        return out
    return out


def compute_dense_total(
    obs_agent: np.ndarray,
    info_agent: Mapping[str, Any],
    prev: Optional[WorldSnapshot],
    terms: List[Tuple[str, Dict[str, Any], float]],
) -> Tuple[float, WorldSnapshot]:
    snap = snapshot_from_info(info_agent)
    total = 0.0
    for name, t_cfg, weight in terms:
        fn = TERM_REGISTRY.get(name)
        if fn is None:
            continue
        total += weight * float(fn(snap, prev, np.asarray(obs_agent, dtype=np.float64), t_cfg))
    return total, snap


def parse_dense_reward_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not cfg or not cfg.get("enabled"):
        return {"enabled": False}
    return {
        "enabled": True,
        "sparse_weight": float(cfg.get("sparse_weight", 1.0)),
        "clip": cfg.get("clip"),
        "terms": _normalize_terms_cfg(cfg.get("terms")),
    }
