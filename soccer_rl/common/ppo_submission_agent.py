"""
Load a Ray RLlib PPO checkpoint for soccer_twos.watch submission agents.

Watch passes per-team observations as {0: obs, 1: obs} (local teammate indices).
Map RLlib Discrete actions to MultiDiscrete branch vectors using ActionFlattener.
"""

from __future__ import annotations

import copy
import inspect
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import ray
import yaml
from gym_unity.envs import ActionFlattener
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface

from soccer_rl.algorithms.ppo.ppo_inference_config import build_ppo_restore_configs
from soccer_rl.common.training_utils import create_rllib_env, load_config


def _find_repo_root(start_dir: str) -> Optional[str]:
    d = os.path.abspath(start_dir)
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "soccer_rl", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    return None


def _resolve_path(agent_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    path = os.path.expanduser(str(path).strip())
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    candidate = os.path.normpath(os.path.join(agent_dir, path))
    return candidate if os.path.exists(candidate) else None


def _parse_team_side(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return 0 if value == 0 else 1
    s = str(value).strip().lower()
    if s in ("blue", "0", "team0", "team_0"):
        return 0
    if s in ("orange", "1", "team1", "team_1"):
        return 1
    return 0


def policy_id_for(team_side: int, local_player_id: int, policy_mode: str) -> str:
    if policy_mode == "team_shared":
        return "team_0" if team_side == 0 else "team_1"
    if policy_mode == "per_player":
        gid = team_side * 2 + int(local_player_id)
        return f"player_{gid}"
    return "default"


def load_agent_yaml(agent_dir: str) -> Dict[str, Any]:
    path = os.path.join(agent_dir, "agent_config.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def build_trainer_for_submission(
    checkpoint_path: str,
    training_config_path: str,
    *,
    base_port: int = 50039,
    fixed_unity_worker_id: int = 77,
) -> Tuple[PPOTrainer, str]:
    """Restore PPOTrainer; returns (trainer, policy_mode from yaml)."""
    cfg = load_config(training_config_path)
    if not cfg:
        raise FileNotFoundError(
            f"Training config missing or empty: {training_config_path}"
        )
    ma_cfg = cfg.get("multiagent") or {}
    policy_mode = ma_cfg.get("policy_mode", "shared_all")

    base_env_config, ppo_template = build_ppo_restore_configs(cfg, base_port)
    ppo_cfg = copy.deepcopy(ppo_template)
    ppo_cfg["env_config"] = {
        **base_env_config,
        "fixed_unity_worker_id": int(fixed_unity_worker_id),
    }

    ray.init(include_dashboard=False, ignore_reinit_error=True)
    tune.registry.register_env("Soccer", create_rllib_env)
    trainer = PPOTrainer(config=ppo_cfg)
    trainer.restore(checkpoint_path)
    return trainer, policy_mode


class PPOSubmissionAgent(AgentInterface):
    """
    RLlib PPO checkpoint runner for soccer_twos.watch.

    Subclasses set ``display_name`` and optionally ``expected_policy_mode`` (warning only).
    """

    display_name = "PPO Submission"
    expected_policy_mode: Optional[str] = None

    def __init__(self, env):
        super().__init__()
        self.name = self.display_name

        agent_dir = os.path.dirname(os.path.abspath(inspect.getfile(type(self))))
        data = load_agent_yaml(agent_dir)

        ckpt = os.environ.get("SOCCER_PPO_CHECKPOINT") or data.get("checkpoint_path")
        cfg_path = (
            os.environ.get("SOCCER_PPO_TRAINING_CONFIG")
            or data.get("training_config_path")
        )

        ckpt = _resolve_path(agent_dir, ckpt)
        cfg_path = _resolve_path(agent_dir, cfg_path)

        if not cfg_path or not os.path.isfile(cfg_path):
            repo = _find_repo_root(agent_dir)
            if repo:
                fallback = os.path.join(
                    repo, "soccer_rl", "algorithms", "ppo", "configs", "config.yaml"
                )
                if os.path.isfile(fallback):
                    cfg_path = fallback

        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(
                "PPO checkpoint not found. Set checkpoint_path in agent_config.yaml "
                "or SOCCER_PPO_CHECKPOINT to the RLlib checkpoint path "
                "(e.g. .../checkpoint_000050/checkpoint-50)."
            )
        if not cfg_path or not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                "Training YAML not found. Set training_config_path in agent_config.yaml "
                "or SOCCER_PPO_TRAINING_CONFIG (must match how you trained)."
            )

        team_side = _parse_team_side(
            os.environ.get("SOCCER_PPO_TEAM_SIDE", data.get("team_side", "blue"))
        )
        base_port = int(
            os.environ.get("SOCCER_BASE_PORT", data.get("base_port", 50039))
        )
        worker_id = int(
            os.environ.get(
                "SOCCER_PPO_UNITY_WORKER_ID",
                data.get("unity_worker_id", 60 + team_side * 35),
            )
        )

        self._flattener = ActionFlattener(env.action_space.nvec)
        self._team_side = team_side
        self._trainer, self._policy_mode = build_trainer_for_submission(
            ckpt,
            cfg_path,
            base_port=base_port,
            fixed_unity_worker_id=worker_id,
        )

        if (
            self.expected_policy_mode
            and self._policy_mode != self.expected_policy_mode
        ):
            print(
                f"[{self.name}] Warning: config policy_mode is {self._policy_mode!r}, "
                f"but this agent expects {self.expected_policy_mode!r}. "
                "Restore may fail or behave incorrectly."
            )

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions: Dict[int, np.ndarray] = {}
        for local_id, obs in observation.items():
            pid = policy_id_for(self._team_side, int(local_id), self._policy_mode)
            flat = self._trainer.compute_action(obs, policy_id=pid)
            vec = self._flattener.lookup_action(flat)
            actions[local_id] = np.asarray(vec, dtype=np.int32)
        return actions
