"""
Team PPO training script for Soccer-Twos.

- Reads run and RLlib options from a YAML config (default: PPO/configs/config.yaml).
- CLI flags override config file values.
- Puts each run in its own timestamped folder under team_runs/ (or config run.output_dir).
- Disables the Ray dashboard (avoids hostname resolution issues on WSL/PACE).
- Saves learning-curve plots every N training iterations.

Usage:
  python PPO/training/train_ppo_team.py
  python PPO/training/train_ppo_team.py --config PPO/configs/config.yaml
  python PPO/training/train_ppo_team.py --max-timesteps 500000

Run from the repository root. On PACE ICE: set PACE_NUM_GPUS if needed.
"""

import argparse
import os
from datetime import datetime

import gym
import yaml
import ray
from ray import tune
from ray.tune import Callback
from ray.rllib import MultiAgentEnv
import soccer_twos
from soccer_twos import EnvType


def create_rllib_env(env_config: dict = None):
    """Create a RLlib-compatible Soccer-Twos env. Used by Ray workers."""
    if env_config is None:
        env_config = {}
    env_config = dict(env_config)
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    if env_config.get("multiagent") is False:
        return env
    class _RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
        pass
    return _RLLibWrapper(env)

# Try matplotlib for saving plots (optional but recommended)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Defaults (used when config file is missing or keys are absent)
NUM_ENVS_PER_WORKER = 3
DEFAULT_PLOT_FREQ = 10
DEFAULT_MAX_TIMESTEPS = 2_000_000
DEFAULT_NUM_WORKERS = 8
BASE_OUTPUT_DIR = "team_runs"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "..", "configs", "config.yaml")


def load_config(path: str) -> dict:
    """Load YAML config. Returns nested dict with 'run', 'resources', 'rllib' keys."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def get_num_gpus():
    """Use 1 GPU by default; allow override via PACE_NUM_GPUS or NUM_GPUS."""
    return int(os.environ.get("PACE_NUM_GPUS", os.environ.get("NUM_GPUS", "1")))


def print_gpu_status(num_gpus_requested: int):
    """Print whether PyTorch sees CUDA and what we're requesting for training."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print()
        print("  GPU status:")
        print(f"    Requested num_gpus for RLlib: {num_gpus_requested}")
        if cuda_available:
            n = torch.cuda.device_count()
            print(f"    PyTorch CUDA: available ({n} device(s))")
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"      [{i}] {name} ({mem:.1f} GiB)")
            if num_gpus_requested == 0:
                print("    -> Training will use CPU (num_gpus=0).")
            else:
                print("    -> Training should use GPU(s) for the learner.")
            print("    Tip: run 'nvidia-smi' in another terminal to monitor GPU use during training.")
        else:
            print("    PyTorch CUDA: not available")
            if num_gpus_requested > 0:
                print("    -> WARNING: You requested GPU but PyTorch has no CUDA. Training will use CPU.")
            else:
                print("    -> Training will use CPU.")
        print()
    except Exception as e:
        print(f"  GPU status: could not check ({e})")
        print()


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


class ProgressPrintCallback(Callback):
    """Prints training progress in a readable table format."""

    def __init__(self, max_timesteps: int):
        self.max_timesteps = max_timesteps
        self._header_printed = set()  # trial_id

    def on_trial_result(self, iteration, trials, trial, result, **info):
        tid = trial.trial_id
        if tid not in self._header_printed:
            self._header_printed.add(tid)
            print()
            print("  iter   timesteps    reward_mean   len_mean   total_time  progress")
            print("  " + "-" * 68)

        ts = result.get("timesteps_total") or 0
        reward = result.get("episode_reward_mean")
        length = result.get("episode_len_mean")
        time_s = result.get("time_total_s") or 0
        progress = min(100.0, 100.0 * ts / self.max_timesteps) if self.max_timesteps else 0

        reward_str = f"{reward:.2f}" if reward is not None else "  -"
        length_str = f"{length:.0f}" if length is not None else "  -"
        time_str = f"{time_s:.0f}s" if time_s else "-"
        print(f"  {iteration:4d}   {ts:10d}   {reward_str:>10}   {length_str:>7}   {time_str:>8}   {progress:5.1f}%")
        print(end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Team PPO training with plots and run folders")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override run.output_dir from config")
    parser.add_argument("--plot-freq", type=int, default=None, help="Override run.plot_freq")
    parser.add_argument("--max-timesteps", type=int, default=None, help="Override run.max_timesteps")
    parser.add_argument("--num-workers", type=int, default=None, help="Override resources.num_workers")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override resources.num_gpus")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_cfg = cfg.get("run") or {}
    res_cfg = cfg.get("resources") or {}
    rllib_cfg = cfg.get("rllib") or {}

    output_dir = args.output_dir or run_cfg.get("output_dir", BASE_OUTPUT_DIR)
    plot_freq = args.plot_freq if args.plot_freq is not None else run_cfg.get("plot_freq", DEFAULT_PLOT_FREQ)
    max_timesteps = args.max_timesteps if args.max_timesteps is not None else run_cfg.get("max_timesteps", DEFAULT_MAX_TIMESTEPS)
    num_workers = args.num_workers if args.num_workers is not None else res_cfg.get("num_workers", DEFAULT_NUM_WORKERS)
    num_envs_per_worker = res_cfg.get("num_envs_per_worker", NUM_ENVS_PER_WORKER)
    checkpoint_freq = run_cfg.get("checkpoint_freq", 100)

    num_gpus = args.num_gpus
    if num_gpus is None:
        from_cfg = res_cfg.get("num_gpus")
        num_gpus = get_num_gpus() if from_cfg is None else int(from_cfg)

    run_name = f"PPO_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Config: {args.config}")
    print(f"Run directory: {os.path.abspath(run_dir)}")
    print(f"Max timesteps: {max_timesteps:,}  (stop when total env steps reach this)")
    print(f"Workers: {num_workers}  |  GPUs: {num_gpus}  |  Plot every: {plot_freq} iters")
    print_gpu_status(num_gpus)
    if not _HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed; plots will not be saved. pip install matplotlib")

    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    plot_callback = PlotCallback(plot_freq=plot_freq)
    progress_callback = ProgressPrintCallback(max_timesteps=max_timesteps)

    # Build RLlib config: code defaults + rllib section from YAML
    model_cfg = rllib_cfg.get("model") or {}
    ppo_config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": rllib_cfg.get("log_level", "INFO"),
        "framework": rllib_cfg.get("framework", "torch"),
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": True,
            "flatten_branched": True,
            "opponent_policy": lambda *_: 0,
        },
        "model": {
            "vf_share_layers": model_cfg.get("vf_share_layers", True),
            "fcnet_hiddens": model_cfg.get("fcnet_hiddens", [512]),
        },
        "rollout_fragment_length": rllib_cfg.get("rollout_fragment_length", 500),
        "train_batch_size": rllib_cfg.get("train_batch_size", 12000),
    }

    analysis = tune.run(
        "PPO",
        name="PPO_Soccer",
        config=ppo_config,
        stop={"timesteps_total": max_timesteps},
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=run_dir,
        callbacks=[plot_callback, progress_callback],
        verbose=0,
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
