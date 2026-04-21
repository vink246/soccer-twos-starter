# Single-player PPO checkpoint agent

**Agent name:** PPO Single (checkpoint)

## Training (how to reproduce the architecture)

Train with this repo’s PPO stack, for example:

`python scripts/train.py --config configs/train_single_ppo_still.yaml`

Use any single-agent config where the learner sees **per-player 336-dim** observations and **Discrete(27)** actions; set `model.hidden_sizes` in the YAML to match **`HIDDEN_SIZES`** in `agent.py` (default **\[256, 256\]**). Policy construction is **self-contained** in **`model.py`**.

At inference, load a PPO checkpoint with matching shapes. For `soccer_twos.watch` (which typically expects MultiDiscrete branched actions), this agent uses `gym_unity.ActionFlattener` to map the flat action index to branched actions.

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

