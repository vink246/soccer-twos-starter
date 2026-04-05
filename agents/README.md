# Agents (submission packages)

Each subdirectory is a Python package implementing `soccer_twos.AgentInterface` for evaluation with:

`python -m soccer_twos.watch -m agents.<package_name>`

After training, copy your best checkpoint (e.g. from `runs/<run>/checkpoints/`) into the agent folder as `checkpoint.pth` (or adjust loading code in your agent).
