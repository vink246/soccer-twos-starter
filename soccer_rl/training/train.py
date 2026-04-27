import argparse
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from soccer_rl.env_factory import make_env
from soccer_rl.training.config_loader import load_experiment_config
from soccer_rl.training.run_layout import prepare_run_dir


def main():
    parser = argparse.ArgumentParser(description="Soccer-Twos PyTorch training")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)

    run_cfg = config.get("run") or {}
    cpu_threads = run_cfg.get("cpu_torch_threads")
    if cpu_threads is not None:
        import torch

        torch.set_num_threads(max(1, int(cpu_threads)))
    run_name = str(run_cfg.get("name", "run"))
    root = str(run_cfg.get("local_dir", "runs"))
    paths = prepare_run_dir(root, run_name)

    env = make_env(config)
    algo = (config.get("algorithm") or {}).get("type", "ppo")
    mod = importlib.import_module("algorithms.%s.trainer" % algo)
    try:
        mod.train(config, env, paths)
    finally:
        env.close()


if __name__ == "__main__":
    main()
