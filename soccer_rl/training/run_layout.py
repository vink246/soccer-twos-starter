import os
from pathlib import Path
from typing import Dict


def prepare_run_dir(root: str, run_name: str) -> Dict[str, Path]:
    base = Path(root) / run_name
    ckpt = base / "checkpoints"
    metrics = base / "metrics"
    plots = base / "plots"
    for p in (ckpt, metrics, plots):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "root": base,
        "checkpoints": ckpt,
        "metrics": metrics,
        "plots": plots,
    }
