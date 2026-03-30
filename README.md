# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Setup

Create and activate the conda environment from the included YAML (Python 3.8 and all dependencies):

```bash
# Clone the repo and enter it
git clone https://github.com/your-github-user/soccer-twos-starter.git
cd soccer-twos-starter

# Create the environment from environment.yml
conda env create -f environment.yml

# Activate it (use the env name from the file, or the name you gave with -n)
conda activate soccertwos
```

If the YAML’s `name:` or `prefix:` point at a specific path (e.g. a shared or home path), create the env with a local name instead:

```bash
conda env create -f environment.yml -n soccertwos
conda activate soccertwos
```

Alternatively, edit `environment.yml` and set `name: soccertwos` (and remove or leave `prefix:`); then `conda env create -f environment.yml` will create an env named `soccertwos`.

To update an existing environment from the YAML:

```bash
conda env update -f environment.yml -n soccertwos --prune
```

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
- **`PPO/training/train_ppo_team.py`** — Entry point. Expects to be run from the **repository root**. Reads the chosen config and starts Ray + RLlib PPO. Imports shared helpers from `training_utils` (see below).

**Shared training utilities (`training_utils.py` in repo root)** — Reusable across PPO, DQN, and other RLlib algorithms: `create_rllib_env` (Soccer-Twos with unique worker_id per Ray worker), `load_config`, `get_num_gpus`, `print_gpu_status`, `PlotCallback`, `ProgressPrintCallback`, `has_matplotlib`. Example for a future DQN script: import these and register the env with `tune.run("DQN", ...)` using the same callbacks.

### Config file (`PPO/configs/config.yaml`)

The YAML has three top-level keys:

| Section      | Purpose |
|-------------|---------|
| `run`       | Output and run control: `output_dir`, `max_timesteps`, `plot_freq`, `checkpoint_freq`. |
| `resources` | Ray/RLlib resources: `num_workers`, `num_gpus`, `num_envs_per_worker`. |
| `rllib`     | Passed into RLlib: `log_level`, `framework`, `model` (e.g. `fcnet_hiddens`), `rollout_fragment_length`, `train_batch_size`. |

Anything not set in the config falls back to script defaults. Env mode and reward shaping are now configurable through YAML (`env` and `multiagent` sections).

### Reward framework (new)

`PPO/rewards/` now provides an env-wrapper reward composer that blends sparse env reward with dense shaping terms:

- `ball_progress` (potential-based ball progress toward opponent goal)
- `ball_goal_distance` (negative distance to opponent goal)
- `trajectory_support` (player proximity to ball-to-goal line)
- `opponent_pressure` (proxy for reducing opponent interception ability)

Configure all reward weights and safeguards in `env.reward`:

- `dense_weight`, `sparse_weight`
- `dense_clip`, `dense_budget_per_episode`
- `goal_reward_dominates` and `goal_reward_threshold`
- `info` fields from the env (`ball_info` / `player_info` positions) for geometry-based terms

This supports both single-agent and multiagent modes.

Quickly inspect per-term reward outputs with a short random rollout:

```bash
python PPO/eval/smoke_test_rewards.py --config PPO/configs/config.yaml --steps 200 --print-every 10
```

### Multiagent policy control modes (new)

You can train with one policy per team or one policy per player using:

- `env.variation` (`multiagent_team` or `multiagent_player`)
- `env.multiagent: true`
- `multiagent.policy_mode`:
  - `team_shared`: one policy controls both teammates (team-level control)
  - `per_player`: each player has its own policy
  - `shared_all`: one shared policy for all 4 players

### Model architecture control (new)

`PPO/training/model_config.py` exposes model presets and pass-through overrides from YAML:

- Presets: `small`, `baseline`, `large`, `residual_mlp`
- Override with any RLlib model keys under `rllib.model` (e.g. `fcnet_hiddens`, `fcnet_activation`, `vf_share_layers`, LSTM/custom model options)

### Playback / rendering (new)

Render checkpoint-vs-checkpoint or checkpoint-vs-random matches:

```bash
python PPO/eval/render_match.py \
  --team-a-checkpoint /path/to/checkpoint-100 \
  --team-b-checkpoint /path/to/checkpoint-100 \
  --policy-mode team_shared \
  --output-video PPO/runs/match.mp4
```

Stop condition defaults to first goal or 120 seconds.

### PACE ICE / HOPPER submission scripts (new)

Two submit helpers are included:

- `scripts/submit_ppo_hopper.sh` (choose `h100` or `h200`)
- `scripts/submit_ppo_a100.sh`

Both request 8-hour jobs and run `PPO/training/train_ppo_team.py`, writing checkpoints and results under the configured run output directory.

Examples:

```bash
# HOPPER H100
bash scripts/submit_ppo_hopper.sh h100 PPO/configs/config.yaml my_exp_h100

# HOPPER H200
bash scripts/submit_ppo_hopper.sh h200 PPO/configs/config.yaml my_exp_h200

# A100
bash scripts/submit_ppo_a100.sh PPO/configs/config.yaml my_exp_a100
```

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

