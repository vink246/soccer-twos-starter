# PPO per-player agent (submission)

**Agent name:** PPOPerPlayer

**Author(s):** *(fill in name & email)*

## Description

Loads a **Ray RLlib PPO** checkpoint trained with **`multiagent.policy_mode: per_player`** (policies `player_0` … `player_3`). For watch, blue team uses `player_0`/`player_1` and orange uses `player_2`/`player_3` (via `team_side`).

## Before you run anything

```bash
conda activate soccertwos
```

## Configure

Edit **`agent_config.yaml`** (same as `ppo_team_agent`: checkpoint path, training YAML, `team_side`). Overrides: `SOCCER_PPO_CHECKPOINT`, `SOCCER_PPO_TRAINING_CONFIG`, `SOCCER_PPO_TEAM_SIDE`.

## Test with watch

```bash
conda activate soccertwos
python -m soccer_twos.watch -m1 ppo_player_agent -m2 example_player_agent
```

Run from **repo root** so `soccer_rl` imports resolve.
