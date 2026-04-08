# Single-player PPO checkpoint agent

**Agent name:** PPO Single (checkpoint)

Loads a PyTorch PPO checkpoint trained with **per-player 336-dim observations** and **Discrete(27)** actions.
For `soccer_twos.watch` (which typically expects MultiDiscrete branched actions), this agent uses
`gym_unity.ActionFlattener` to convert the flat action index to branched actions.

## Setup

Copy your weights into this folder as **`checkpoint.pth`** (e.g. from `runs/<run>/checkpoints/checkpoint_final.pth`).

Optional override:

```bash
export SINGLE_PPO_CHECKPOINT=/path/to/checkpoint.pth
```

## Watch

Against CEIA baseline:

```bash
python -m soccer_twos.watch -m1 agents.single_ppo_checkpoint_agent -m2 agents.ceia_baseline_agent
```

Self-play:

```bash
python -m soccer_twos.watch -m agents.single_ppo_checkpoint_agent
```

