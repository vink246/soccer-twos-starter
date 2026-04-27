from typing import Any, Dict

import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        h = list(hidden_sizes)
        self.shared = mlp([obs_dim] + h, activation=nn.Tanh, output_activation=nn.Tanh)
        self.pi_logits = nn.Linear(h[-1], n_actions)
        self.v = nn.Linear(h[-1], 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        logits = self.pi_logits(x)
        value = self.v(x).squeeze(-1)
        return logits, value


def build_model(architecture: str, obs_dim: int, n_actions: int, model_cfg: Dict[str, Any]) -> nn.Module:
    if architecture != "mlp_actor_critic":
        raise ValueError(
            "Standalone checkpoint agent only supports architecture='mlp_actor_critic'; got %r"
            % architecture
        )
    hidden = tuple(model_cfg.get("hidden_sizes", [256, 256]))
    return MLPActorCritic(obs_dim, n_actions, hidden_sizes=hidden)
