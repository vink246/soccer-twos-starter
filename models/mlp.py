import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """Discrete policy + value head for PPO (single-agent)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        h = list(hidden_sizes)
        body = mlp([obs_dim] + h, activation=nn.Tanh, output_activation=nn.Tanh)
        self.shared = body
        self.pi_logits = nn.Linear(h[-1], n_actions)
        self.v = nn.Linear(h[-1], 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        logits = self.pi_logits(x)
        value = self.v(x).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value


class MLPQNetwork(nn.Module):
    """DQN-style Q(s, a_all) for discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        h = list(hidden_sizes)
        self.net = mlp([obs_dim] + h + [n_actions], activation=nn.ReLU, output_activation=None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MLPDoubleQDiscrete(nn.Module):
    """Twin Q-networks for discrete SAC-style training."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = MLPQNetwork(obs_dim, n_actions, hidden_sizes)
        self.q2 = MLPQNetwork(obs_dim, n_actions, hidden_sizes)

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)


class MLPCategoricalActor(nn.Module):
    """Policy network outputting logits over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        h = list(hidden_sizes)
        self.net = mlp([obs_dim] + h + [n_actions], activation=nn.ReLU, output_activation=None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def sample(self, obs: torch.Tensor):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, dist.entropy()
