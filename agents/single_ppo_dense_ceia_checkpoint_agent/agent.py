"""
AgentInterface for checkpoints from configs/train_single_ppo_dense_vs_ceia_self_teammate.yaml.

Same network as generic single-player PPO: 336-dim obs, Discrete(27), mlp_actor_critic [256,256].
"""

import os
from typing import Dict

import gym
import numpy as np
import torch
from gym_unity.envs import ActionFlattener

from .model import build_model
from soccer_twos import AgentInterface

OBS_DIM = 336
N_ACTIONS = 27
ARCHITECTURE = "mlp_actor_critic"
HIDDEN_SIZES = [256, 256]

_CHECKPOINT_ENV = "SINGLE_PPO_DENSE_CEIA_CHECKPOINT"


class SinglePPODenseCeiaCheckpointAgent(AgentInterface):
    """Loads a PPO .pth and acts for both team players with the same policy (watch)."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "PPO Single dense vs CEIA (checkpoint)"
        self.env = env

        override = os.environ.get(_CHECKPOINT_ENV, "").strip()
        weights_path = (
            override
            if override
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth")
        )

        model_cfg = {"architecture": ARCHITECTURE, "hidden_sizes": HIDDEN_SIZES}
        self.model = build_model(ARCHITECTURE, OBS_DIM, N_ACTIONS, model_cfg)
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
        else:
            raise FileNotFoundError(
                "Missing %s. Copy e.g. runs/ppo_single_dense_vs_ceia_self_teammate/checkpoints/checkpoint_final.pth "
                "to agents/single_ppo_dense_ceia_checkpoint_agent/checkpoint.pth or set %s."
                % (weights_path, _CHECKPOINT_ENV)
            )
        self.model.eval()

        self._use_flattener = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        if self._use_flattener:
            self.flattener = ActionFlattener(env.action_space.nvec)
        else:
            self.flattener = None

    def _act_one(self, obs: np.ndarray) -> int:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.size != OBS_DIM:
            raise ValueError("Expected obs_dim=%d, got %d" % (OBS_DIM, o.size))
        with torch.no_grad():
            t = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model.forward(t)
            return int(torch.argmax(logits, dim=-1).item())

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        keys = sorted(observation.keys())
        if len(keys) != 2:
            raise ValueError("Expected two player observations (keys 0 and 1); got %s" % keys)

        a0 = self._act_one(observation[keys[0]])
        a1 = self._act_one(observation[keys[1]])

        if self._use_flattener:
            return {keys[0]: self.flattener.lookup_action(a0), keys[1]: self.flattener.lookup_action(a1)}
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return {
                keys[0]: np.asarray(a0, dtype=np.int64),
                keys[1]: np.asarray(a1, dtype=np.int64),
            }
        raise TypeError(
            "Unsupported action_space %s; use default watch env (MultiDiscrete) or Discrete(27)."
            % type(self.env.action_space)
        )
