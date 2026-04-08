# Agents (submission packages)

Each subdirectory is a Python package implementing `soccer_twos.AgentInterface` for evaluation with:

`python -m soccer_twos.watch -m agents.<package_name>`

After training, copy your best checkpoint (e.g. from `runs/<run>/checkpoints/`) into the agent folder as `checkpoint.pth` (or adjust loading code in your agent).

For **team PPO** checkpoints trained with `configs/train_team_ppo_dense_vs_random.yaml`, use [`team_ppo_dense_agent/`](team_ppo_dense_agent/) (joint 729 actions, 672-dim concat obs).

For **single-player PPO** checkpoints (per-player 27 actions), use [`single_ppo_checkpoint_agent/`](single_ppo_checkpoint_agent/).
