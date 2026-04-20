import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.registry import build_model
from soccer_rl.env_factory import build_make_kwargs, install_opponent_policy_on_env
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


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_val = last_value
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_val = values[t]
    ret = adv + values
    return adv, ret


def train(config: Dict[str, Any], env, run_paths: Dict[str, Path]) -> None:
    run_cfg = config.get("run") or {}
    seed = int(run_cfg.get("seed", 0))
    set_seed(seed)

    device_spec = config.get("device", "auto")
    from soccer_rl.training.device import resolve_device

    device = resolve_device(str(device_spec))

    algo = config.get("algorithm") or {}
    rollout_cfg = config.get("rollout") or {}
    n_steps = int(rollout_cfg.get("n_steps", algo.get("n_steps", 2048)))

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
    arch = model_cfg.get("architecture", "mlp_actor_critic")
    model = build_model(arch, obs_dim, n_act, model_cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=float(algo.get("lr", 3e-4)))
    if single and bool(tm.get("self_teammate", False)):
        # TeamVsPolicyWrapper supports swapping teammate policy at runtime. This closure
        # reads current model weights, so the teammate mirrors the learner as it trains.
        deterministic_tm = bool(tm.get("self_teammate_deterministic", True))

        def _self_teammate_policy(obs_tm: Any, *_args: Any, **_kwargs: Any) -> int:
            o = torch.as_tensor(obs_tm, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_tm, _, _ = model.act(o, deterministic=deterministic_tm)
            return int(a_tm.item())

        setter = getattr(env, "set_teammate_policy", None)
        if callable(setter):
            setter(_self_teammate_policy)

    opponent_model = None
    self_play_sync_every: Optional[int] = None
    self_play_opp_phase: Optional[Dict[str, Any]] = None
    if single and bool(tm.get("self_play_opponent", False)):
        n_sync = int(tm.get("self_play_update_every_iterations", 50))
        if n_sync < 1:
            raise ValueError("training_mode.self_play_update_every_iterations must be >= 1")
        self_play_sync_every = n_sync
        opp_det = bool(tm.get("self_play_opponent_deterministic", True))

        initial_spec = tm.get("self_play_opponent_initial")
        initial_opp_fn: Any = None
        if initial_spec is not None:
            if isinstance(initial_spec, str):
                initial_opp_fn = build_policy(initial_spec, team_action_space)
            elif isinstance(initial_spec, dict):
                initial_opp_fn = build_actor_policy_from_spec(
                    initial_spec, team_action_space
                )
            elif callable(initial_spec):
                initial_opp_fn = initial_spec
            else:
                raise TypeError(
                    "training_mode.self_play_opponent_initial must be str, dict, or callable"
                )

        opponent_model = build_model(arch, obs_dim, n_act, model_cfg).to(device)
        if initial_opp_fn is None:
            opponent_model.load_state_dict(model.state_dict())
        opponent_model.eval()

        self_play_opp_phase = {"use_initial": initial_opp_fn is not None}

        def _coerce_discrete_action(a: Any) -> int:
            return int(np.asarray(a, dtype=np.int64).reshape(-1)[0])

        def _self_play_opponent_policy(obs_opp: Any, *_args: Any, **_kwargs: Any) -> int:
            if self_play_opp_phase["use_initial"] and initial_opp_fn is not None:
                return _coerce_discrete_action(initial_opp_fn(obs_opp))
            o = torch.as_tensor(obs_opp, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_opp, _, _ = opponent_model.act(o, deterministic=opp_det)
            return int(a_opp.item())

        if not install_opponent_policy_on_env(env, _self_play_opponent_policy):
            raise RuntimeError(
                "self_play_opponent is enabled but no set_opponent_policy was found on the "
                "env chain; use team_vs_policy with single_player."
            )

    gamma = float(algo.get("gamma", 0.99))
    lam = float(algo.get("gae_lambda", 0.95))
    clip = float(algo.get("clip_range", 0.2))
    vf_coef = float(algo.get("vf_coef", 0.5))
    ent_coef = float(algo.get("ent_coef", 0.01))
    max_grad_norm = float(algo.get("max_grad_norm", 0.5))
    n_epochs = int(algo.get("n_epochs", 4))
    batch_size = int(algo.get("batch_size", 256))

    metrics_path = run_paths["metrics"] / "training_log.csv"
    logger = MetricsLogger(
        metrics_path,
        [
            "timestep",
            "iteration",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "episode_return_mean",
            "goals_per_episode_mean",
            "sparse_episode_return_mean",
            "sparse_goals_per_episode_mean",
        ],
    )

    timestep = 0
    iteration = 0
    episode_index = 0
    ep_returns: deque = deque(maxlen=50)
    ep_goals: deque = deque(maxlen=50)
    ep_sparse_returns: deque = deque(maxlen=50)
    ep_sparse_goals: deque = deque(maxlen=50)
    cur_rewards: List[float] = []
    cur_sparse_rewards: List[float] = []

    plot_every_episodes = int(run_cfg.get("plot_every_episodes", 0))
    checkpoint_every_episodes = int(run_cfg.get("checkpoint_every_episodes", 0))

    goal_thr = float((config.get("metrics") or {}).get("goal_reward_threshold", 0.25))
    sparse_goal_thr = float(
        (config.get("metrics") or {}).get("sparse_goal_reward_threshold", goal_thr)
    )

    def finish_episode() -> None:
        if not cur_rewards:
            return
        ep_returns.append(float(np.sum(cur_rewards)))
        ep_goals.append(float(episode_goal_estimate(cur_rewards, threshold=goal_thr)))
        cur_rewards.clear()
        if cur_sparse_rewards:
            ep_sparse_returns.append(float(np.sum(cur_sparse_rewards)))
            ep_sparse_goals.append(
                float(episode_goal_estimate(cur_sparse_rewards, threshold=sparse_goal_thr))
            )
            cur_sparse_rewards.clear()

    while timestep < max_steps:
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(n_steps):
            if single:
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, logp, val = model.act(o_t)
                a_int = int(action.item())
                next_obs, reward, done, _info = env.step(a_int)
                cur_rewards.append(float(reward))
                sparse_r = float(
                    (_info.get("_dense_reward") or {}).get("sparse_reward", float(reward))
                ) if isinstance(_info, dict) else float(reward)
                cur_sparse_rewards.append(sparse_r)
            else:
                o_t = torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, logp, val = model.act(o_t)
                a0 = int(action.item())
                a1 = int(opp_team_fn(obs[1]))
                if isinstance(env.action_space, gym.spaces.Discrete):
                    next_obs, reward, done, _info = env.step(
                        np.array([a0, a1], dtype=np.int64)
                    )
                else:
                    next_obs, reward, done, _info = env.step({0: a0, 1: a1})
                cur_rewards.append(float(reward[0]))
                sparse_r = float(reward[0])
                if isinstance(_info, dict):
                    dense_meta = _info.get("_dense_reward")
                    if isinstance(dense_meta, dict):
                        sparse_r = float(
                            (dense_meta.get(0) or {}).get("sparse_reward", sparse_r)
                        )
                cur_sparse_rewards.append(sparse_r)
                done = bool(done["__all__"])

            obs_buf.append(np.asarray(obs if single else obs[0], dtype=np.float32))
            act_buf.append(a_int if single else a0)
            rew_buf.append(float(reward if single else reward[0]))
            val_buf.append(float(val.item()))
            logp_buf.append(float(logp.item()))
            done_buf.append(bool(done))

            timestep += 1
            obs = next_obs
            if (single and done) or (not single and done):
                finish_episode()
                episode_index += 1
                obs = env.reset()
                if plot_every_episodes > 0 and episode_index % plot_every_episodes == 0:
                    plot_training_csv(metrics_path, run_paths["plots"], x_key="timestep")
                    plot_training_csv(
                        metrics_path,
                        run_paths["plots"],
                        x_key="timestep",
                        y_keys=["goals_per_episode_mean"],
                    )
                if checkpoint_every_episodes > 0 and episode_index % checkpoint_every_episodes == 0:
                    ep_path = (
                        run_paths["checkpoints"]
                        / f"checkpoint_episode_{episode_index}.pth"
                    )
                    torch.save(model.state_dict(), ep_path)
            if timestep >= max_steps:
                break

        with torch.no_grad():
            if single:
                last_o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                last_o = torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
            _, last_v = model.forward(last_o)
            last_v = float(last_v.item())

        rewards = np.asarray(rew_buf, dtype=np.float32)
        values = np.asarray(val_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        adv, rets = _compute_gae(rewards, values, dones, last_v, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_buf, dtype=torch.long, device=device)
        logp_old = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(rets, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)

        n = obs_t.shape[0]
        idxs = np.arange(n)
        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        total_loss = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, batch_size):
                mb = idxs[start : start + batch_size]
                if len(mb) == 0:
                    continue
                ob = obs_t[mb]
                ac = act_t[mb]
                logp_o = logp_old[mb]
                ad = adv_t[mb]
                rt = ret_t[mb]

                logp, ent, v = model.evaluate(ob, ac)
                ratio = torch.exp(logp - logp_o)
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * ad
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(v, rt)
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent.mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

                total_pi_loss += float(pi_loss.item())
                total_v_loss += float(v_loss.item())
                total_ent += float(ent.mean().item())
                total_loss += float(loss.item())
                n_updates += 1

        iteration += 1
        if opponent_model is not None and self_play_sync_every is not None:
            if iteration % self_play_sync_every == 0:
                opponent_model.load_state_dict(model.state_dict())
                if self_play_opp_phase is not None and self_play_opp_phase["use_initial"]:
                    self_play_opp_phase["use_initial"] = False

        if n_updates == 0:
            n_updates = 1
        ep_mean = float(np.mean(ep_returns)) if ep_returns else 0.0
        g_mean = float(np.mean(ep_goals)) if ep_goals else 0.0
        ep_sparse_mean = float(np.mean(ep_sparse_returns)) if ep_sparse_returns else 0.0
        g_sparse_mean = float(np.mean(ep_sparse_goals)) if ep_sparse_goals else 0.0
        logger.log(
            {
                "timestep": timestep,
                "iteration": iteration,
                "loss": total_loss / n_updates,
                "policy_loss": total_pi_loss / n_updates,
                "value_loss": total_v_loss / n_updates,
                "entropy": total_ent / n_updates,
                "episode_return_mean": ep_mean,
                "goals_per_episode_mean": g_mean,
                "sparse_episode_return_mean": ep_sparse_mean,
                "sparse_goals_per_episode_mean": g_sparse_mean,
            }
        )

        ckpt_every = int(run_cfg.get("checkpoint_every_iterations", 5))
        if iteration % ckpt_every == 0:
            ckpt_path = run_paths["checkpoints"] / f"checkpoint_iter_{iteration}.pth"
            torch.save(model.state_dict(), ckpt_path)
            if bool(run_cfg.get("plot_on_checkpoint", False)):
                plot_training_csv(metrics_path, run_paths["plots"], x_key="timestep")
                plot_training_csv(
                    metrics_path,
                    run_paths["plots"],
                    x_key="timestep",
                    y_keys=["goals_per_episode_mean"],
                )

    finish_episode()
    logger.close()
    final = run_paths["checkpoints"] / "checkpoint_final.pth"
    torch.save(model.state_dict(), final)

    plot_training_csv(metrics_path, run_paths["plots"], x_key="timestep")
    plot_training_csv(
        metrics_path,
        run_paths["plots"],
        x_key="timestep",
        y_keys=["goals_per_episode_mean"],
    )
