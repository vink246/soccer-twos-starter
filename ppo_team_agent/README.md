# PPO team agent (submission)

**Agent name:** PPOTeamShared

**Author(s):** *(fill in name & email)*

## Description

Loads a **Ray RLlib PPO** checkpoint trained with **`multiagent.policy_mode: team_shared`** (one policy per team: `team_0`, `team_1`). Use this package with `python -m soccer_twos.watch` after training (see root **README**).

## Before you run anything

Activate the conda environment (dependencies include `soccer_twos`, Ray, Unity):

```bash
conda activate soccertwos
```

## Configure

1. Edit **`agent_config.yaml`**: set `checkpoint_path` to your RLlib checkpoint (directory that `trainer.restore(...)` accepts, e.g. `.../checkpoint_000100/checkpoint-100`).
2. Set `training_config_path` to the **exact** YAML used for training (or rely on the default lookup from the repo root).
3. For **orange** team in `-m1`/`-m2` mode, set `team_side: orange` in the yaml **or** export `SOCCER_PPO_TEAM_SIDE=orange`.

You can override paths with environment variables: `SOCCER_PPO_CHECKPOINT`, `SOCCER_PPO_TRAINING_CONFIG`, `SOCCER_PPO_TEAM_SIDE`.

## Test with watch

From the **repository root** (so `soccer_rl` is importable):

```bash
conda activate soccertwos
python -m soccer_twos.watch -m1 ppo_team_agent -m2 example_player_agent
```

## Zip for submission

Zip this folder only if graders can import `soccer_rl` from the same project layout, **or** vendor minimal code. For course submission, submitting the **whole repo** with this folder is usually simplest.
