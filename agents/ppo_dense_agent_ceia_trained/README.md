# Single-player PPO (dense vs CEIA + self-teammate) checkpoint agent (512x512)

**Agent name:** PPO dense CEIA trained (checkpoint)

Loads weights trained with `configs/train_single_ppo_dense_vs_ceia_self_teammate.yaml` using hidden sizes `[512, 512]`:

- **Obs:** 336 per player
- **Action:** Discrete(27) (flattened), mapped through `ActionFlattener` for `soccer_twos.watch`

## Setup

Copy your best checkpoint to **`checkpoint.pth`** in this folder, e.g. from:

`runs/ppo_single_dense_vs_ceia_self_teammate/checkpoints/checkpoint_final.pth`

Override path:

```bash
export PPO_DENSE_AGENT_CEIA_TRAINED_CHECKPOINT=/path/to/checkpoint.pth
```

## Watch

```bash
python -m soccer_twos.watch -m1 agents.ppo_dense_agent_ceia_trained -m2 agents.ceia_baseline_agent
```
