# Single-player PPO dense self-play checkpoint agent (512×512)

**Agent name:** PPO dense CEIA trained (checkpoint)

Loads weights from a **dense single-agent PPO** run with **`mlp_actor_critic` hidden sizes `[512, 512]`** (not the `[256, 256]` run in `train_single_ppo_dense_vs_ceia_self_teammate.yaml`). Representative training configs in this repo:

- [`configs/train_single_ppo_dense_self_play_10m.yaml`](../../configs/train_single_ppo_dense_self_play_10m.yaml) — long self-play / CEIA-style setup with **512×512** trunk.
- [`configs/train_single_ppo_dense_vs_ppo_dense_agent_ceia_trained.yaml`](../../configs/train_single_ppo_dense_vs_ppo_dense_agent_ceia_trained.yaml) — continues training with a **frozen opponent** loaded from this folder’s `checkpoint.pth` (same 512×512 head for the learner).

Inference:

- **Obs:** 336 per player
- **Action:** Discrete(27) (flattened), mapped through `ActionFlattener` for `soccer_twos.watch`

Policy code is **self-contained** in **`model.py`** (no repo `models/` import).

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
