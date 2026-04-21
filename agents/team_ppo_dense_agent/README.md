# Team PPO checkpoint agent (dense vs random training config)

**Agent name:** PPO Team (dense vs random)

**Description**

## Training

```bash
python scripts/train.py --config configs/train_team_ppo_dense_vs_random.yaml
```

**PPO** on the **team** policy: **672**-dim observation (two 336-dim players concatenated), **729** joint discrete actions, **vs random** opponent team, **dense reward shaping**, **`mlp_actor_critic` with hidden \[256, 256\]**. Policy code is **self-contained** in **`model.py`**.

Loads a PyTorch policy trained with the above (or any compatible **multiagent_team** setup with `flatten_branched: true`, same obs/action layout). Implements `soccer_twos.AgentInterface` for `soccer_twos.watch`.

## Setup

1. Copy your best weights into this folder as **`checkpoint.pth`** (e.g. from `runs/ppo_team_dense_vs_random/checkpoints/checkpoint_final.pth`).
2. This package is **self-contained**: policy construction lives in **`model.py`** next to `agent.py` (no dependency on the repo’s top-level `models/` package).

Optional: point to a weights file elsewhere:

```bash
export TEAM_PPO_DENSE_CHECKPOINT=/path/to/your.pth
python -m soccer_twos.watch -m1 agents.team_ppo_dense_agent -m2 agents.ceia_baseline_agent
```

## Watch

```bash
python -m soccer_twos.watch -m1 agents.team_ppo_dense_agent -m2 agents.ceia_baseline_agent
```

Self-play (same policy both teams):

```bash
python -m soccer_twos.watch -m agents.team_ppo_dense_agent
```

## Notes

- Inference uses **argmax** (deterministic) on the joint action distribution.
- If you change **hidden sizes** or architecture in training, update the constants at the top of `agent.py` to match.
