from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RewardContext:
    """Context passed to reward terms each step."""

    prev_obs: Optional[Dict[int, Any]]
    obs: Dict[int, Any]
    base_reward: Dict[int, float]
    info: Dict[int, Any]
    done: Dict[int, bool]
    step_idx: int
    cfg: Dict[str, Any]


class RewardTerm:
    """Interface for a single dense reward term."""

    name = "reward_term"

    def compute(self, ctx: RewardContext) -> Dict[int, float]:
        raise NotImplementedError


class RewardComposer:
    """
    Combine weighted reward terms into per-agent shaped rewards and team summaries.
    """

    def __init__(self, terms: List[RewardTerm], cfg: Dict[str, Any]):
        self.terms = terms
        self.cfg = cfg or {}
        self._dense_budget_used = 0.0

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def compose(self, ctx: RewardContext) -> Dict[str, Any]:
        per_term_cfg = self.cfg.get("terms", {})
        term_contribs: Dict[str, Dict[int, float]] = {}
        dense_total: Dict[int, float] = {aid: 0.0 for aid in ctx.obs.keys()}

        for term in self.terms:
            tcfg = per_term_cfg.get(term.name, {})
            if not tcfg.get("enabled", True):
                continue
            weight = self._to_float(tcfg.get("weight", 1.0), default=1.0)
            raw = term.compute(ctx)
            contrib: Dict[int, float] = {}
            for aid in dense_total.keys():
                value = self._to_float(raw.get(aid, 0.0))
                contrib[aid] = weight * value
                dense_total[aid] += contrib[aid]
            term_contribs[term.name] = contrib

        global_clip = self.cfg.get("dense_clip", {})
        clip_enabled = bool(global_clip.get("enabled", True))
        clip_min = self._to_float(global_clip.get("min", -1.0))
        clip_max = self._to_float(global_clip.get("max", 1.0))
        if clip_enabled:
            dense_total = {
                aid: self._clip(val, clip_min, clip_max) for aid, val in dense_total.items()
            }

        max_dense_budget = self.cfg.get("dense_budget_per_episode")
        if max_dense_budget is not None:
            max_dense_budget = abs(self._to_float(max_dense_budget, default=0.0))
            if max_dense_budget > 0:
                remaining = max(0.0, max_dense_budget - self._dense_budget_used)
                abs_dense = sum(abs(v) for v in dense_total.values())
                if abs_dense > 0 and abs_dense > remaining:
                    scale = remaining / abs_dense
                    dense_total = {aid: val * scale for aid, val in dense_total.items()}
                self._dense_budget_used += sum(abs(v) for v in dense_total.values())

        sparse_weight = self._to_float(self.cfg.get("sparse_weight", 1.0), default=1.0)
        dense_weight = self._to_float(self.cfg.get("dense_weight", 0.05), default=0.05)
        warmup_steps = int(self.cfg.get("dense_warmup_steps", 0))
        if ctx.step_idx < warmup_steps:
            dense_weight = 0.0

        final = {}
        for aid in ctx.obs.keys():
            base_val = self._to_float(ctx.base_reward.get(aid, 0.0))
            final_val = sparse_weight * base_val + dense_weight * dense_total.get(aid, 0.0)
            final[aid] = final_val

        term_goal_threshold = self._to_float(self.cfg.get("goal_reward_threshold", 0.9), default=0.9)
        goal_override = bool(self.cfg.get("goal_reward_dominates", True))
        if goal_override:
            for aid in ctx.obs.keys():
                base_val = self._to_float(ctx.base_reward.get(aid, 0.0))
                if abs(base_val) >= term_goal_threshold:
                    final[aid] = sparse_weight * base_val

        team_map = self.cfg.get("team_map", {})
        teams = {"team_0": [0, 1], "team_1": [2, 3]}
        if isinstance(team_map, dict) and team_map:
            teams = {
                key: [int(a) for a in val]
                for key, val in team_map.items()
                if isinstance(val, list)
            }
        team_rewards = {
            team: sum(final.get(aid, 0.0) for aid in aids) for team, aids in teams.items()
        }

        return {
            "per_agent": final,
            "dense_per_agent": dense_total,
            "per_term": term_contribs,
            "team_rewards": team_rewards,
        }

    def reset(self) -> None:
        self._dense_budget_used = 0.0
