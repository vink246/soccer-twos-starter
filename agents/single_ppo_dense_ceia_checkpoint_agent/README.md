# Single-player PPO (dense vs CEIA + self-teammate) checkpoint agent

**Agent name:** PPO Single dense vs CEIA (checkpoint)

Loads weights trained with `configs/train_single_ppo_dense_vs_ceia_self_teammate.yaml`:

- **Obs:** 336 per player  
- **Action:** Discrete(27) (flattened), mapped through `ActionFlattener` for `soccer_twos.watch`

## Setup

Copy your best checkpoint to **`checkpoint.pth`** in this folder, e.g. from:

`runs/ppo_single_dense_vs_ceia_self_teammate/checkpoints/checkpoint_final.pth`

Override path:

```bash
export SINGLE_PPO_DENSE_CEIA_CHECKPOINT=/path/to/checkpoint.pth
```

## Watch

```bash
python -m soccer_twos.watch -m1 agents.single_ppo_dense_ceia_checkpoint_agent -m2 agents.ceia_baseline_agent
```

If you change `hidden_sizes` or architecture in training, update the constants at the top of `agent.py`.
