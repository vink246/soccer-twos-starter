# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 6. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.


## PPO

This section describes the PPO (Proximal Policy Optimization) training setup: layout, config file, and how to run it.

### Overview

- **Algorithm**: PPO via [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html). The policy is a neural network (MLP) that maps observations to actions.
- **Setting**: Single-agent Soccer-Twos — one controlled agent vs a fixed opponent (e.g. always action 0). The env is `team_vs_policy` with `single_player=True` and flattened observations/actions.
- **Run layout**: Each training run gets its own timestamped directory. Inside it, Ray writes checkpoints, and our callbacks write learning-curve plots and a progress table to the console.
- **Config**: Run and RLlib options are read from a YAML config file so you can maintain different configs per experiment (e.g. short run vs long run). CLI flags override the config file.

### Folder structure

```
PPO/
├── configs/
│   └── config.yaml      # Default config (run, resources, rllib)
└── training/
    └── train_ppo_team.py   # Training script (run from repo root)
```

- **`PPO/configs/`** — YAML configs. Use `config.yaml` as the default; copy it to create per-run configs (e.g. `config_fast.yaml`, `config_long.yaml`) and pass with `--config PPO/configs/config_fast.yaml`.
- **`PPO/training/train_ppo_team.py`** — Entry point. Expects to be run from the **repository root** (so `utils` and `soccer_twos` resolve). Reads the chosen config and starts Ray + RLlib PPO.

### Config file (`PPO/configs/config.yaml`)

The YAML has three top-level keys:

| Section      | Purpose |
|-------------|---------|
| `run`       | Output and run control: `output_dir`, `max_timesteps`, `plot_freq`, `checkpoint_freq`. |
| `resources` | Ray/RLlib resources: `num_workers`, `num_gpus`, `num_envs_per_worker`. |
| `rllib`     | Passed into RLlib: `log_level`, `framework`, `model` (e.g. `fcnet_hiddens`), `rollout_fragment_length`, `train_batch_size`. |

Anything not set in the config falls back to script defaults. The env (e.g. `opponent_policy`) is fixed in code and is not configurable via YAML.

### How to run

From the **repository root**:

```bash
# Default config (PPO/configs/config.yaml)
python PPO/training/train_ppo_team.py

# Custom config for this run
python PPO/training/train_ppo_team.py --config PPO/configs/config_fast.yaml

# Override options from the command line (override config file)
python PPO/training/train_ppo_team.py --max-timesteps 500000 --output-dir my_runs --num-gpus 0
```

**CLI flags** (all override the config file):

| Flag | Overrides | Example |
|------|-----------|--------|
| `--config` | Which YAML to load | `--config PPO/configs/config.yaml` |
| `--output-dir` | `run.output_dir` | `--output-dir my_runs` |
| `--max-timesteps` | `run.max_timesteps` | `--max-timesteps 1000000` |
| `--plot-freq` | `run.plot_freq` | `--plot-freq 5` |
| `--num-workers` | `resources.num_workers` | `--num-workers 4` |
| `--num-gpus` | `resources.num_gpus` | `--num-gpus 0` (CPU only) |

### Outputs

- **Run directory**: `{output_dir}/PPO_team_{timestamp}/` (e.g. `team_runs/PPO_team_20260308_120000/`). Set `output_dir` in the config or with `--output-dir`.
- **Checkpoints**: Under the run directory, in the Ray trial folder (e.g. `.../PPO_Soccer_xxx/checkpoint_000100/checkpoint-100`). Used for resuming or for the watch script.
- **Plots**: In the trial’s `plots/` subfolder — learning-curve images saved every `plot_freq` iterations.
- **Console**: A progress table (iteration, timesteps, reward mean, progress %) and, at startup, GPU status and the config path.

### GPU and PACE

- At startup the script prints whether PyTorch sees CUDA and how many GPUs you requested. Use `--num-gpus 0` for CPU-only if you hit CUDA “no kernel image” errors.
- On PACE ICE, set `PACE_NUM_GPUS` (or `NUM_GPUS`) in the job environment if you want to override the config’s `num_gpus` without passing `--num-gpus`.


## Team-specific additions

This section describes additions made by the team for training and running on shared infrastructure (e.g. PACE ICE).

### PPO training (`PPO/training/train_ppo_team.py`)

The main PPO training entry point is described in the **PPO** section above. In short:

- **Config**: Options are read from `PPO/configs/config.yaml` (or a file you pass with `--config`). You can create one config per run (e.g. `config_fast.yaml`, `config_long.yaml`).
- **Run from repo root**: `python PPO/training/train_ppo_team.py` or `python PPO/training/train_ppo_team.py --config PPO/configs/your_config.yaml --max-timesteps 500000`.
- **Outputs**: Each run gets a timestamped folder under `team_runs/` (or the `output_dir` in your config). Checkpoints and plots are written there; the script prints a progress table and GPU status.
- **Dashboard**: Uses `ray.init(include_dashboard=False)` to avoid hostname/socket issues on WSL and PACE.
- **GPU**: Use `--num-gpus 0` for CPU-only if you see CUDA "no kernel image" errors. See the PPO section for more detail.

