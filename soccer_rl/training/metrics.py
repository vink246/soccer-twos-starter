import csv
from pathlib import Path
from typing import Any, Dict, List


class MetricsLogger:
    """Append-only CSV metrics for training iterations."""

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        new_file = not path.is_file()
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        if new_file:
            self._w.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def episode_goal_estimate(episode_rewards: List[float], threshold: float = 0.25) -> int:
    """Heuristic goal count from per-step rewards (tune threshold for your reward scale)."""
    return sum(1 for r in episode_rewards if abs(float(r)) >= threshold)
