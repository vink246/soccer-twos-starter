"""Load a frozen PyTorch policy from disk for teammate / opponent callables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import gym
import numpy as np
import torch
from torch.distributions import Categorical

from models.registry import build_model

_ACTOR_ARCHS = frozenset({"mlp_actor_critic", "mlp_categorical_actor"})


def build_actor_policy_from_spec(
    spec: Dict[str, Any], action_space: gym.spaces.Discrete
) -> Callable:
    """
    Build a callable(obs, *a, **kw) -> int action index for TeamVsPolicyWrapper or trainers.

    ``spec`` example::

        kind: checkpoint
        path: runs/foo/checkpoint_final.pth
        obs_dim: 336
        deterministic: true
        model:
          architecture: mlp_actor_critic
          hidden_sizes: [256, 256]
    """
    if not isinstance(spec, dict):
        raise TypeError("checkpoint spec must be a dict")
    kind = str(spec.get("kind", "checkpoint")).lower()
    if kind != "checkpoint":
        raise ValueError("policy checkpoint spec requires kind: checkpoint")

    path = Path(spec["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("checkpoint path not found: %s" % path)

    obs_dim = int(spec["obs_dim"])
    n_act = int(action_space.n)
    model_cfg = dict(spec.get("model") or {})
    arch = str(model_cfg.get("architecture", "mlp_actor_critic"))
    if arch not in _ACTOR_ARCHS:
        raise ValueError(
            "checkpoint policies support architectures %s, got %r"
            % (sorted(_ACTOR_ARCHS), arch)
        )

    model = build_model(arch, obs_dim, n_act, model_cfg)
    state = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    deterministic = bool(spec.get("deterministic", True))

    def policy(obs: Any, *_args: Any, **_kwargs: Any) -> int:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.size != obs_dim:
            raise ValueError(
                "checkpoint policy expected obs_dim=%d, got %d values"
                % (obs_dim, o.size)
            )
        t = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if arch == "mlp_actor_critic":
                logits, _ = model(t)
            else:
                logits = model(t)
            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    return policy
