"""
AgentInterface for policies trained with configs like train_team_ppo_dense_vs_random.yaml:
multiagent_team, flatten_branched=true, joint Discrete(729), obs dim 672 (2 x 336).
"""

import os
from typing import Dict

import gym
import numpy as np
import torch
from gym_unity.envs import ActionFlattener

from .model import build_model
from soccer_twos import AgentInterface

# Must match train_team_ppo_dense_vs_random.yaml + env_factory inference
OBS_DIM = 336 * 2
N_ACTIONS = 729  # 27 * 27 joint
ACTIONS_PER_PLAYER = 27
ARCHITECTURE = "mlp_actor_critic"
HIDDEN_SIZES = [256, 256]

_CHECKPOINT_ENV = "TEAM_PPO_DENSE_CHECKPOINT"


class TeamPPODenseCheckpointAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "PPO Team (dense vs random)"
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
                "Missing %s. Copy e.g. runs/ppo_team_dense_vs_random/checkpoints/checkpoint_final.pth "
                "to agents/team_ppo_dense_agent/checkpoint.pth or set %s."
                % (weights_path, _CHECKPOINT_ENV)
            )
        self.model.eval()

        self._use_flattener = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        if self._use_flattener:
            self.flattener = ActionFlattener(env.action_space.nvec)
        else:
            self.flattener = None

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        keys = sorted(observation.keys())
        if len(keys) != 2:
            raise ValueError(
                "Expected observations for two team players (keys 0 and 1); got keys %s" % keys
            )
        o0 = np.asarray(observation[keys[0]], dtype=np.float32).reshape(-1)
        o1 = np.asarray(observation[keys[1]], dtype=np.float32).reshape(-1)
        if o0.size != 336 or o1.size != 336:
            raise ValueError(
                "Expected two 336-dim player vectors; got sizes %d and %d" % (o0.size, o1.size)
            )

        obs = np.concatenate([o0, o1], axis=0)
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model.forward(t)
            joint = int(torch.argmax(logits, dim=-1).item())

        a0 = joint // ACTIONS_PER_PLAYER
        a1 = joint % ACTIONS_PER_PLAYER

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
