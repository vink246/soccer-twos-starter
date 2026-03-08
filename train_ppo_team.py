"""
Team PPO training script for Soccer-Twos.

- Puts each run in its own timestamped folder under team_runs/.
- Disables the Ray dashboard (avoids hostname resolution issues on WSL/PACE).
- Saves learning-curve plots every N training iterations.
- Uses GPU when available; respects PACE/CLUSTER env for resource hints.

Usage:
  python train_ppo_team.py
  python train_ppo_team.py --plot-freq 5 --max-timesteps 500000
  python train_ppo_team.py --output-dir my_runs

On PACE ICE (HPC): run via your job script; set PACE_NUM_GPUS if needed.
"""

import argparse
import os
from datetime import datetime

import ray
from ray import tune
from ray.tune import Callback
from soccer_twos import EnvType

from utils import create_rllib_env

# Try matplotlib for saving plots (optional but recommended)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Defaults
NUM_ENVS_PER_WORKER = 3
DEFAULT_PLOT_FREQ = 10
DEFAULT_MAX_TIMESTEPS = 2_000_000
DEFAULT_NUM_WORKERS = 8
BASE_OUTPUT_DIR = "team_runs"


def get_num_gpus():
    """Use 1 GPU by default; allow override via PACE_NUM_GPUS or NUM_GPUS."""
    return int(os.environ.get("PACE_NUM_GPUS", os.environ.get("NUM_GPUS", "1")))


class PlotCallback(Callback):
    """Saves reward/timesteps plots every N training iterations."""

    def __init__(self, plot_freq: int = 10):
        self.plot_freq = plot_freq
        self._history = {}  # trial_id -> list of result dicts

    def on_trial_result(self, iteration, trials, trial, result, **info):
        tid = trial.trial_id
        if tid not in self._history:
            self._history[tid] = []
        self._history[tid].append({
            "iteration": iteration,
            "timesteps_total": result.get("timesteps_total") or result.get("timesteps_this_iter", 0),
            "episode_reward_mean": result.get("episode_reward_mean"),
            "episode_len_mean": result.get("episode_len_mean"),
        })
        # Accumulate timesteps if not reported
        hist = self._history[tid]
        if len(hist) > 1 and hist[-1]["timesteps_total"] == 0:
            prev = hist[-2].get("timesteps_total") or 0
            hist[-1]["timesteps_total"] = prev + result.get("timesteps_this_iter", 0)

        if not _HAS_MATPLOTLIB or iteration % self.plot_freq != 0 or iteration == 0:
            return

        logdir = getattr(trial, "logdir", None) or getattr(trial, "local_dir", None)
        if not logdir:
            logdir = result.get("logdir")
        if not logdir:
            return
        plot_dir = os.path.join(logdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Build x-axis: prefer timesteps_total, else use iteration
        timesteps = [r.get("timesteps_total") for r in hist]
        if all(t is not None and t > 0 for t in timesteps):
            x_vals = timesteps
            x_label = "Timesteps"
        else:
            x_vals = [r["iteration"] for r in hist]
            x_label = "Training iteration"
        rewards = [r.get("episode_reward_mean") for r in hist if r.get("episode_reward_mean") is not None]
        if not rewards:
            return
        n = min(len(x_vals), len(rewards))
        x_vals, rewards = x_vals[:n], rewards[:n]

        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(x_vals, rewards, color="C0", alpha=0.7)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Episode reward mean")
            ax.set_title(f"PPO learning curve (iter {iteration})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(plot_dir, f"reward_curve_iter_{iteration:05d}.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"[PlotCallback] Failed to save plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Team PPO training with plots and run folders")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=BASE_OUTPUT_DIR,
        help=f"Base directory for run folders (default: {BASE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--plot-freq",
        type=int,
        default=DEFAULT_PLOT_FREQ,
        help=f"Save a plot every N training iterations (default: {DEFAULT_PLOT_FREQ})",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=DEFAULT_MAX_TIMESTEPS,
        help=f"Stop after this many env timesteps (default: {DEFAULT_MAX_TIMESTEPS})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of Ray workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: 1, or PACE_NUM_GPUS / NUM_GPUS env)",
    )
    args = parser.parse_args()

    num_gpus = args.num_gpus if args.num_gpus is not None else get_num_gpus()
    run_name = f"PPO_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {os.path.abspath(run_dir)}")
    if not _HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed; plots will not be saved. pip install matplotlib")

    # Dashboard disabled to avoid hostname resolution issues (WSL / PACE head nodes)
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    plot_callback = PlotCallback(plot_freq=args.plot_freq)

    analysis = tune.run(
        "PPO",
        name="PPO_Soccer",
        config={
            "num_gpus": num_gpus,
            "num_workers": args.num_workers,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={"timesteps_total": args.max_timesteps},
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir=run_dir,
        callbacks=[plot_callback],
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial:
        print("Best trial:", best_trial)
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print("Best checkpoint:", best_checkpoint)
    print("Done. Results in:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
