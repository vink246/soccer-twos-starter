from typing import Any, Dict

import torch.nn as nn

from models.mlp import MLPActorCritic, MLPCategoricalActor, MLPDoubleQDiscrete, MLPQNetwork

_REGISTRY = {
    "mlp_actor_critic": MLPActorCritic,
    "mlp_q": MLPQNetwork,
    "mlp_double_q": MLPDoubleQDiscrete,
    "mlp_categorical_actor": MLPCategoricalActor,
}


def build_model(
    architecture: str, obs_dim: int, n_actions: int, model_cfg: Dict[str, Any]
) -> nn.Module:
    cls = _REGISTRY.get(architecture)
    if cls is None:
        raise ValueError(f"Unknown model architecture: {architecture!r}. Known: {list(_REGISTRY)}")

    hidden = tuple(model_cfg.get("hidden_sizes", [256, 256]))
    if cls is MLPActorCritic:
        return cls(obs_dim, n_actions, hidden_sizes=hidden)
    if cls is MLPQNetwork or cls is MLPCategoricalActor:
        return cls(obs_dim, n_actions, hidden_sizes=hidden)
    if cls is MLPDoubleQDiscrete:
        return cls(obs_dim, n_actions, hidden_sizes=hidden)
    raise TypeError(cls)
