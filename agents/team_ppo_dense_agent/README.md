# Team PPO checkpoint agent (dense vs random training config)

**Agent name:** PPO Team (dense vs random)

**Description**

Loads a PyTorch policy trained with `configs/train_team_ppo_dense_vs_random.yaml` (or any **multiagent_team** setup with `flatten_branched: true`, **672**-dim team observation, **729** joint actions). Implements `soccer_twos.AgentInterface` for `soccer_twos.watch`.

## Setup

1. Copy your best weights into this folder as **`checkpoint.pth`** (e.g. from `runs/ppo_team_dense_vs_random/checkpoints/checkpoint_final.pth`).
2. Run training or watch from the **repository root** so `models` imports resolve (`PYTHONPATH=.` is default when you `cd` to the repo).

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
