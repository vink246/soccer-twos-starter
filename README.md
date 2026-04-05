# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

```bash
git clone https://github.com/your-github-user/soccer-twos-starter.git
cd soccer-twos-starter
```

### 2. Create and activate conda environment

```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
```

### 3. Downgrade build tools for compatibility

```bash
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Fix protobuf and pydantic compatibility

```bash
pip install protobuf==3.20.3
pip install pydantic==1.10.13
```

### 6. Try a visual rollout (packaged agent)

With the Soccer-Twos Unity binary available:

```bash
python -m soccer_twos.watch -m agents.example_player_agent
```

For a minimal **multi-agent random** loop against the env API (no AgentInterface package), use the script under [`archived/`](archived/):

```bash
python archived/example_random_players.py
```

### 7. Train

**Recommended:** PyTorch trainers and YAML configs (see [Usage (PyTorch training)](#usage-pytorch-training) below):

```bash
python scripts/train.py --config configs/train_single_ppo_still.yaml
```

**Legacy Ray RLlib** examples live under [`archived/`](archived/); they need `PYTHONPATH=.` from the repo root (see [`archived/README.md`](archived/README.md)).

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Example agents live under the [`agents/`](agents/) directory (see [`agents/example_player_agent/`](agents/example_player_agent/) and [`agents/example_team_agent/`](agents/example_team_agent/)).
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent description and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.).

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m agents.example_player_agent`

Similarly, you can test your own agent by replacing the module path with your package under `agents/` (e.g. `agents.my_agent`).

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, extract the `ceia_baseline_agent` folder into this project (for example under [`agents/ceia_baseline_agent/`](agents/ceia_baseline_agent/) so it can be imported as `agents.ceia_baseline_agent`). For instance you can run,

`python -m soccer_twos.watch -m1 agents.example_player_agent -m2 agents.ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.


---

# Team Modifications

This section describes the **team modifications**: project layout, PyTorch training, configs, and `runs/` outputs. The assignment instructions in the first part of this README still apply; this stack is an additional path alongside the legacy Ray examples in [`archived/`](archived/). For visual rollouts, use **`python -m soccer_twos.watch`** (see [Testing/Evaluating](#testingevaluating) above).

## Project structure

```text
soccer-twos-starter/
  agents/                    # Submission-style AgentInterface packages (zip from here)
    example_player_agent/
    example_team_agent/
    README.md
  soccer_rl/                 # Shared training / env utilities
    env_factory.py           # soccer_twos.make + YAML policy preset resolution
    policy_presets.py        # e.g. still, random (teammate / opponent / opponent team)
    training/
      train.py               # CLI entry (see Usage)
      config_loader.py
      device.py
      run_layout.py
      metrics.py
      plotting.py
  algorithms/                # One folder per RL method
    ppo/   (trainer.py, defaults.yaml)
    dqn/
    sac/
  models/                    # PyTorch nn.Modules + registry
    mlp.py
    registry.py
  configs/                   # Experiment YAML files
  runs/                      # Training outputs (gitignored): checkpoints, metrics, plots
  scripts/
    train.py                 # Thin launcher for training
    soccerstwos_job.batch    # Example SLURM job (PACE-style)
  archived/                  # Legacy Ray RLlib + old root-level examples (see archived/README.md)
  utils.py                   # RLLib env helper used by archived Ray scripts
  environment-information.md # Env shapes, rewards, ports (reference notes)
```

Legacy Ray RLlib scripts are under [`archived/`](archived/) and import [`utils.py`](utils.py). New work can use only the PyTorch pipeline above and ignore Ray unless you need those examples.

## Usage (PyTorch training)

From the **repository root**, with the Soccer-Twos Unity binary available:

```bash
python -m soccer_rl.training.train --config configs/train_single_ppo_still.yaml
```

or:

```bash
python scripts/train.py --config configs/train_single_dqn_random.yaml
```

Example configs in [`configs/`](configs/):

| Config | Purpose |
|--------|---------|
| `train_single_ppo_still.yaml` | Single agent, `team_vs_policy`, still teammate/opponent |
| `train_single_dqn_random.yaml` | Single agent, random opponent preset |
| `train_single_ppo_vs_checkpoint_example.yaml` | Single agent vs a frozen policy checkpoint |
| `train_team_ppo.yaml` | One policy vs opponent team, `multiagent_team` |
| `train_team_ppo_vs_checkpoint_example.yaml` | Team mode vs frozen opponent checkpoint |
| `train_ceia_baseline_ppo_selfplay.yaml` | Team-mode reference aligned with CEIA baseline notes |
| `train_single_sac.yaml` | Discrete SAC-style training, single agent |

YAML is merged with algorithm defaults from `algorithms/<type>/defaults.yaml`. Key sections:

- **`run`**: `name`, `local_dir` (default `runs`), `seed`, `stop.total_timesteps`, checkpoint frequency.
- **`device`**: `auto`, `cpu`, or `cuda` / `cuda:0` (networks on GPU; Unity stepping stays on CPU).
- **`training_mode`**: `single_agent: true|false`; `teammate_policy` / `opponent_policy` (single agent); `opponent_team_policy` (team mode).
- **`env`**: `flatten_branched`, `base_port`, `worker_id`, etc.
- **`algorithm`**: `type: ppo | dqn | sac` plus method-specific hyperparameters.
- **`model`**: `architecture` and `hidden_sizes` (see [`models/registry.py`](models/registry.py)).

## Training outputs (`runs/`)

Each run writes under `runs/<run_name>/`:

- **`checkpoints/`** — `checkpoint_final.pth` (and periodic saves). DQN/PPO use a single `state_dict`; SAC stores `{"policy": ..., "q": ...}`.
- **`metrics/training_log.csv`** — Scalars over training (returns, losses, heuristic goal counts).
- **`plots/`** — PNG curves from the CSV (e.g. episode return, goals heuristic).

`runs/` is listed in [`.gitignore`](.gitignore).

## Packaging a trained agent

After training, copy the checkpoint you want into your agent package (for example `agents/my_agent/checkpoint.pth`) and load it inside your `AgentInterface` implementation, following the pattern in [`agents/example_team_agent/`](agents/example_team_agent/). See [`agents/README.md`](agents/README.md) for a short checklist.

## Extra dependencies

PyTorch training and plots add packages listed in [requirements.txt](requirements.txt) (e.g. `torch`, `PyYAML`, `matplotlib`) on top of the original Soccer-Twos / ML-Agents stack.

## Environment reference

For observation shapes, action spaces, reward aggregation, and GRPC ports, see [environment-information.md](environment-information.md).

