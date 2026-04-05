import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_experiment_config(
    config_path: Path,
    algorithm_defaults_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load main YAML and merge algorithm defaults from algorithms/<type>/defaults.yaml
    unless algorithm_defaults_path is provided.
    """
    cfg = load_yaml(config_path)
    algo = (cfg.get("algorithm") or {}).get("type", "ppo")
    repo_root = Path(__file__).resolve().parents[2]
    if algorithm_defaults_path is None:
        algorithm_defaults_path = repo_root / "algorithms" / str(algo) / "defaults.yaml"
    if algorithm_defaults_path.is_file():
        defaults = load_yaml(algorithm_defaults_path)
        cfg = deep_merge(defaults, cfg)
    return cfg
