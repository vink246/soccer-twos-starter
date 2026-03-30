"""
Shared utilities for RLlib training scripts (PPO, DQN, etc.).

- create_rllib_env: Soccer-Twos env with unique worker_id per Ray worker (for Unity ports).
- load_config: Load YAML config file.
- get_num_gpus / print_gpu_status: GPU detection and status.
- PlotCallback / ProgressPrintCallback: Tune callbacks for plots and progress table.

Use from repo root: python PPO/training/train_ppo_team.py (adds root to path if needed).
"""

import os

import gym
from ray.rllib import MultiAgentEnv
from ray.tune import Callback
import soccer_twos

try:
    from PPO.rewards import RewardShapingWrapper
except Exception:
    RewardShapingWrapper = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None


def create_rllib_env(env_config=None):
    """Create a RLlib-compatible Soccer-Twos env. Used by Ray workers.
    RLlib passes an EnvContext (dict-like) with worker_index, vector_index, and env config.
    Each worker must get a unique worker_id so Unity ports don't collide.
    """
    context = env_config if env_config is not None else {}
    if hasattr(context, "keys") and callable(getattr(context, "keys", None)):
        config_dict = dict(context)
    elif hasattr(context, "config") and isinstance(getattr(context, "config", None), dict):
        config_dict = dict(context.config.get("env_config", {}))
    else:
        config_dict = {}
    if hasattr(context, "worker_index") and hasattr(context, "vector_index"):
        num_envs_per_worker = config_dict.get("num_envs_per_worker", 1)
        worker_id = context.worker_index * num_envs_per_worker + context.vector_index
        config_dict["worker_id"] = worker_id
    reward_cfg = config_dict.pop("reward", None)
    env = soccer_twos.make(**config_dict)
    if (
        isinstance(reward_cfg, dict)
        and reward_cfg.get("enabled", False)
        and RewardShapingWrapper is not None
    ):
        env = RewardShapingWrapper(env, reward_cfg)
    if config_dict.get("multiagent") is False:
        return env
    class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
        pass
    return RLLibWrapper(env)


def load_config(path: str) -> dict:
    """Load YAML config file. Returns nested dict (e.g. run, resources, rllib)."""
    if not path or not os.path.isfile(path):
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required for load_config; pip install pyyaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def get_num_gpus():
    """Number of GPUs to use (from PACE_NUM_GPUS or NUM_GPUS env, default 1)."""
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
    """Tune callback: save reward/timesteps plot every N training iterations."""

    def __init__(self, plot_freq: int = 10):
        super().__init__()
        self.plot_freq = plot_freq
        self._history = {}

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
        hist = self._history[tid]
        if len(hist) > 1 and hist[-1]["timesteps_total"] == 0:
            prev = hist[-2].get("timesteps_total") or 0
            hist[-1]["timesteps_total"] = prev + result.get("timesteps_this_iter", 0)

        if not _HAS_MATPLOTLIB or iteration % self.plot_freq != 0 or iteration == 0:
            return

        logdir = getattr(trial, "logdir", None) or getattr(trial, "local_dir", None) or result.get("logdir")
        if not logdir:
            return
        plot_dir = os.path.join(logdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        timesteps = [r.get("timesteps_total") for r in hist]
        if all(t is not None and t > 0 for t in timesteps):
            x_vals, x_label = timesteps, "Timesteps"
        else:
            x_vals, x_label = [r["iteration"] for r in hist], "Training iteration"
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
            ax.set_title(f"Learning curve (iter {iteration})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"reward_curve_iter_{iteration:05d}.png"), dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"[PlotCallback] Failed to save plot: {e}")


class ProgressPrintCallback(Callback):
    """Tune callback: print training progress table (iter, timesteps, reward, progress)."""

    def __init__(self, max_timesteps: int):
        super().__init__()
        self.max_timesteps = max_timesteps
        self._header_printed = set()

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


def has_matplotlib():
    """Whether matplotlib is available for PlotCallback."""
    return _HAS_MATPLOTLIB
