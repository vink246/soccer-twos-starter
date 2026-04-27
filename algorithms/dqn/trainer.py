import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

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
        self.obs_dim = obs_dim
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
    q = build_model(model_cfg.get("architecture", "mlp_q"), obs_dim, n_act, model_cfg).to(
        device
    )
    target = build_model(model_cfg.get("architecture", "mlp_q"), obs_dim, n_act, model_cfg).to(
        device
    )
    target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=float(algo.get("lr", 1e-4)))

    gamma = float(algo.get("gamma", 0.99))
    batch_size = int(algo.get("batch_size", 64))
    buffer_size = int(algo.get("buffer_size", 50_000))
    learning_starts = int(algo.get("learning_starts", 1000))
    train_freq = int(algo.get("train_freq", 4))
    target_update_freq = int(algo.get("target_update_freq", 1000))
    eps_start = float(algo.get("epsilon_start", 1.0))
    eps_end = float(algo.get("epsilon_end", 0.05))
    eps_decay = int(algo.get("epsilon_decay_steps", 50_000))

    buf = ReplayBuffer(buffer_size, obs_dim)
    goal_thr = float((config.get("metrics") or {}).get("goal_reward_threshold", 0.25))

    metrics_path = run_paths["metrics"] / "training_log.csv"
    logger = MetricsLogger(
        metrics_path,
        [
            "timestep",
            "loss",
            "epsilon",
            "episode_return_mean",
            "goals_per_episode_mean",
        ],
    )

    from collections import deque as dq

    ep_returns: dq = dq(maxlen=50)
    ep_goals: dq = dq(maxlen=50)
    cur_rewards: List[float] = []

    def finish_episode() -> None:
        if not cur_rewards:
            return
        ep_returns.append(float(np.sum(cur_rewards)))
        ep_goals.append(float(episode_goal_estimate(cur_rewards, threshold=goal_thr)))
        cur_rewards.clear()

    timestep = 0
    loss_val = 0.0

    while timestep < max_steps:
        eps = eps_end + max(0.0, (eps_start - eps_end) * (1.0 - timestep / max(1, eps_decay)))

        if single:
            if random.random() < eps:
                a = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    qt = q(
                        torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    )
                    a = int(torch.argmax(qt, dim=-1).item())
            next_obs, reward, done, _ = env.step(a)
            cur_rewards.append(float(reward))
            o = np.asarray(obs, dtype=np.float32).reshape(-1)
            o2 = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            buf.add(o, a, float(reward), o2, float(done))
        else:
            if random.random() < eps:
                a0 = int(np.random.randint(0, n_act))
            else:
                with torch.no_grad():
                    qt = q(
                        torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
                    )
                    a0 = int(torch.argmax(qt, dim=-1).item())
            a1 = int(opp_team_fn(obs[1]))
            if isinstance(env.action_space, gym.spaces.Discrete):
                next_obs, reward, done, _ = env.step(np.array([a0, a1], dtype=np.int64))
            else:
                next_obs, reward, done, _ = env.step({0: a0, 1: a1})
            cur_rewards.append(float(reward[0]))
            done = bool(done["__all__"])
            o = np.asarray(obs[0], dtype=np.float32).reshape(-1)
            o2 = np.asarray(next_obs[0], dtype=np.float32).reshape(-1)
            buf.add(o, a0, float(reward[0]), o2, float(done))

        timestep += 1
        obs = next_obs

        if (single and done) or (not single and done):
            finish_episode()
            obs = env.reset()

        if buf.size >= learning_starts and timestep % train_freq == 0:
            ob, ac, rw, nx, dn = buf.sample(batch_size, device)
            with torch.no_grad():
                q_next = target(nx).max(dim=-1)[0]
                target_q = rw + (1.0 - dn) * gamma * q_next
            q_sa = q(ob).gather(1, ac.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_sa, target_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())

        if timestep % target_update_freq == 0 and timestep > 0:
            target.load_state_dict(q.state_dict())

        if timestep % 500 == 0:
            ep_mean = float(np.mean(ep_returns)) if ep_returns else 0.0
            g_mean = float(np.mean(ep_goals)) if ep_goals else 0.0
            logger.log(
                {
                    "timestep": timestep,
                    "loss": loss_val,
                    "epsilon": eps,
                    "episode_return_mean": ep_mean,
                    "goals_per_episode_mean": g_mean,
                }
            )

        ckpt_every = int(run_cfg.get("checkpoint_every_steps", 20_000))
        if timestep % ckpt_every == 0 and timestep > 0:
            torch.save(q.state_dict(), run_paths["checkpoints"] / f"checkpoint_{timestep}.pth")

    finish_episode()
    logger.close()
    torch.save(q.state_dict(), run_paths["checkpoints"] / "checkpoint_final.pth")
    plot_training_csv(metrics_path, run_paths["plots"], x_key="timestep")
    plot_training_csv(
        metrics_path,
        run_paths["plots"],
        x_key="timestep",
        y_keys=["goals_per_episode_mean"],
    )
