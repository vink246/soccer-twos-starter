from pathlib import Path
from typing import Optional

import csv


def plot_training_csv(
    csv_path: Path,
    plots_dir: Path,
    x_key: str = "timestep",
    y_keys: Optional[list] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not csv_path.is_file():
        return

    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return

    y_keys = y_keys or ["episode_return_mean", "loss"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    for yk in y_keys:
        xs, ys = [], []
        for row in rows:
            if yk not in row or x_key not in row:
                continue
            try:
                xv = float(row[x_key])
                yv = float(row[yk])
            except (TypeError, ValueError):
                continue
            xs.append(xv)
            ys.append(yv)
        if len(xs) < 2:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, linewidth=1)
        plt.xlabel(x_key)
        plt.ylabel(yk)
        plt.title(yk.replace("_", " "))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = plots_dir / f"{yk}.png"
        plt.savefig(out, dpi=120)
        plt.close()
