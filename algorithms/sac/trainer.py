import copy
import random
from pathlib import Path
from typing import Any, Dict, List

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.mlp import MLPCategoricalActor, MLPDoubleQDiscrete
from models.registry import build_model
from soccer_rl.env_factory import build_make_kwargs
from soccer_rl.policy_checkpoint import build_actor_policy_from_spec
from soccer_rl.policy_presets import build_policy
from soccer_rl.training.metrics import MetricsLogger, episode_goal_estimate
from soccer_rl.training.plotting import plot_training_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.buf_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.buf_next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.buf_a = np.zeros((capacity,), dtype=np.int64)
        self.buf_r = np.zeros((capacity,), dtype=np.float32)
        self.buf_d = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, o, a, r, o2, d):
        self.buf_obs[self.idx] = o
        self.buf_next[self.idx] = o2
        self.buf_a[self.idx] = a
        self.buf_r[self.idx] = r
        self.buf_d[self.idx] = d
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.as_tensor(self.buf_obs[idx], device=device),
            torch.as_tensor(self.buf_a[idx], device=device),
            torch.as_tensor(self.buf_r[idx], device=device),
            torch.as_tensor(self.buf_next[idx], device=device),
            torch.as_tensor(self.buf_d[idx], device=device),
        )


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def train(config: Dict[str, Any], env, run_paths: Dict[str, Path]) -> None:
    run_cfg = config.get("run") or {}
    set_seed(int(run_cfg.get("seed", 0)))

    from soccer_rl.training.device import resolve_device

    device = resolve_device(str(config.get("device", "auto")))

    algo = config.get("algorithm") or {}
    stop = run_cfg.get("stop") or {}
    max_steps = int(stop.get("total_timesteps", 100_000))

    single = (config.get("training_mode") or {}).get("single_agent", True)
    _, team_action_space = build_make_kwargs(config)
    tm = config.get("training_mode") or {}
    opp_team = tm.get("opponent_team_policy", "random")
    if isinstance(opp_team, str):
        opp_team_fn = build_policy(opp_team, team_action_space)
    elif isinstance(opp_team, dict):
        opp_team_fn = build_actor_policy_from_spec(opp_team, team_action_space)
    elif callable(opp_team):
        opp_team_fn = opp_team
    else:
        raise TypeError("opponent_team_policy must be str, dict, or callable")

    obs0 = env.reset()
    if single:
        obs_dim = int(np.asarray(obs0, dtype=np.float64).size)
        n_act = int(env.action_space.n)
    else:
        obs_dim = int(np.asarray(obs0[0], dtype=np.float64).size)
        ac = env.action_space
        if isinstance(ac, gym.spaces.Discrete):
            n_act = int(ac.n)
        elif isinstance(ac, gym.spaces.Dict):
            n_act = int(ac.spaces[0].n)
        elif isinstance(ac, gym.spaces.Tuple):
            n_act = int(ac.spaces[0].n)
        else:
            n_act = int(ac[0].n)

    obs = obs0
    model_cfg = config.get("model") or {}
    hidden = tuple(model_cfg.get("hidden_sizes", [256, 256]))

    actor_arch = model_cfg.get("actor_architecture", "mlp_categorical_actor")
    critic_arch = model_cfg.get("critic_architecture", "mlp_double_q")

    if actor_arch == "mlp_categorical_actor":
        policy = MLPCategoricalActor(obs_dim, n_act, hidden_sizes=hidden).to(device)
    else:
        policy = build_model(actor_arch, obs_dim, n_act, model_cfg).to(device)

    if critic_arch == "mlp_double_q":
        q = MLPDoubleQDiscrete(obs_dim, n_act, hidden_sizes=hidden).to(device)
        q_targ = copy.deepcopy(q).to(device)
        for p in q_targ.parameters():
            p.requires_grad = False
    else:
        raise ValueError("SAC expects mlp_double_q critic")

    opt_q = optim.Adam(q.parameters(), lr=float(algo.get("lr_q", 3e-4)))
    opt_pi = optim.Adam(policy.parameters(), lr=float(algo.get("lr_pi", 3e-4)))

    gamma = float(algo.get("gamma", 0.99))
    alpha = float(algo.get("alpha", 0.2))
    batch_size = int(algo.get("batch_size", 256))
    buffer_size = int(algo.get("buffer_size", 100_000))
    learning_starts = int(algo.get("learning_starts", 5000))
    train_freq = int(algo.get("train_freq", 1))
    tau = float(algo.get("tau", 0.005))

    buf = ReplayBuffer(buffer_size, obs_dim)
    goal_thr = float((config.get("metrics") or {}).get("goal_reward_threshold", 0.25))

    metrics_path = run_paths["metrics"] / "training_log.csv"
    logger = MetricsLogger(
        metrics_path,
        ["timestep", "loss_q", "loss_pi", "episode_return_mean", "goals_per_episode_mean"],
    )

    from collections import deque

    ep_returns: deque = deque(maxlen=50)
    ep_goals: deque = deque(maxlen=50)
    cur_rewards: List[float] = []

    def finish_episode() -> None:
        if not cur_rewards:
            return
        ep_returns.append(float(np.sum(cur_rewards)))
        ep_goals.append(float(episode_goal_estimate(cur_rewards, threshold=goal_thr)))
        cur_rewards.clear()

    timestep = 0
    loss_q_val = 0.0
    loss_pi_val = 0.0

    while timestep < max_steps:
        with torch.no_grad():
            if single:
                logits = policy(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                )
            else:
                logits = policy(
                    torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
                )
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())

        if single:
            next_obs, reward, done, _ = env.step(a)
            cur_rewards.append(float(reward))
            o = np.asarray(obs, dtype=np.float32).reshape(-1)
            o2 = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            buf.add(o, a, float(reward), o2, float(done))
        else:
            a1 = int(opp_team_fn(obs[1]))
            if isinstance(env.action_space, gym.spaces.Discrete):
                next_obs, reward, done, _ = env.step(np.array([a, a1], dtype=np.int64))
            else:
                next_obs, reward, done, _ = env.step({0: a, 1: a1})
            cur_rewards.append(float(reward[0]))
            done = bool(done["__all__"])
            o = np.asarray(obs[0], dtype=np.float32).reshape(-1)
            o2 = np.asarray(next_obs[0], dtype=np.float32).reshape(-1)
            buf.add(o, a, float(reward[0]), o2, float(done))

        timestep += 1
        obs = next_obs

        if (single and done) or (not single and done):
            finish_episode()
            obs = env.reset()

        if buf.size >= learning_starts and timestep % train_freq == 0:
            ob, ac, rw, nx, dn = buf.sample(batch_size, device)

            with torch.no_grad():
                logits_n = policy(nx)
                log_pi_n = F.log_softmax(logits_n, dim=-1)
                pi_n = log_pi_n.exp()
                q1n, q2n = q_targ(nx)
                q_min_n = torch.min(q1n, q2n)
                v_next = (pi_n * (q_min_n - alpha * log_pi_n)).sum(dim=-1)
                target_q = rw + (1.0 - dn) * gamma * v_next

            q1, q2 = q(ob)
            q1_a = q1.gather(1, ac.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, ac.unsqueeze(1)).squeeze(1)
            loss_q = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            loss_q_val = float(loss_q.item())

            logits_p = policy(ob)
            log_pi = F.log_softmax(logits_p, dim=-1)
            pi = log_pi.exp()
            q1p, q2p = q(ob)
            q_min = torch.min(q1p, q2p)
            loss_pi = (pi * (alpha * log_pi - q_min.detach())).sum(dim=-1).mean()
            opt_pi.zero_grad()
            loss_pi.backward()
            opt_pi.step()
            loss_pi_val = float(loss_pi.item())

            soft_update(q_targ, q, tau)

        if timestep % 500 == 0:
            ep_mean = float(np.mean(ep_returns)) if ep_returns else 0.0
            g_mean = float(np.mean(ep_goals)) if ep_goals else 0.0
            logger.log(
                {
                    "timestep": timestep,
                    "loss_q": loss_q_val,
                    "loss_pi": loss_pi_val,
                    "episode_return_mean": ep_mean,
                    "goals_per_episode_mean": g_mean,
                }
            )

        ckpt_every = int(run_cfg.get("checkpoint_every_steps", 20_000))
        if timestep % ckpt_every == 0 and timestep > 0:
            ck = run_paths["checkpoints"] / f"checkpoint_{timestep}.pth"
            torch.save({"policy": policy.state_dict(), "q": q.state_dict()}, ck)

    finish_episode()
    logger.close()
    torch.save(
        {"policy": policy.state_dict(), "q": q.state_dict()},
        run_paths["checkpoints"] / "checkpoint_final.pth",
    )
    plot_training_csv(metrics_path, run_paths["plots"], x_key="timestep")
    plot_training_csv(
        metrics_path,
        run_paths["plots"],
        x_key="timestep",
        y_keys=["goals_per_episode_mean"],
    )
