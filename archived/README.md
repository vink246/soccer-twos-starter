# Archived Ray / legacy examples

These scripts predate the PyTorch training stack in [`soccer_rl/`](../soccer_rl/) and [`algorithms/`](../algorithms/). They depend on **Ray RLlib** (see [requirements.txt](../requirements.txt)) and on [`utils.py`](../utils.py) for `create_rllib_env`.

Run them from the **repository root** so imports resolve:

```bash
cd /path/to/soccer-twos-starter
PYTHONPATH=. python archived/example_ray_ppo_sp_still.py
```

Other entry points in this folder include `example_random_players.py`, `example_random_teams.py`, `example_ray_*.py`, and `train_ray_*.py`.
