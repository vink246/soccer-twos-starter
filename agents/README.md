# Agents (submission packages)

Each subdirectory is a Python package implementing `soccer_twos.AgentInterface` for evaluation with:

`python -m soccer_twos.watch -m agents.<package_name>`

## Self-contained checkpoint agents

The PyTorch checkpoint agents (`single_ppo_checkpoint_agent`, `single_ppo_dense_ceia_checkpoint_agent`, `ppo_dense_agent_ceia_trained`, `team_ppo_dense_agent`) ship a local **`model.py`** in the same folder. It defines the **`mlp_actor_critic`** network (`MLPActorCritic` + Tanh MLP) and a **`build_model`** helper so the package does **not** depend on the repository’s top-level `models/` package. When you zip a submission, include **`agent.py`**, **`model.py`**, **`__init__.py`**, and **`checkpoint.pth`** (and any `README.md` you maintain).

## How these models were trained

Training uses this repo’s PyTorch stack:

```bash
python scripts/train.py --config configs/<config_name>.yaml
```

Checkpoints are written under `runs/<run_name>/checkpoints/` (see each YAML’s `run:` section). All of the checkpoint agents below use **PPO** (`algorithm.type: ppo`) and **`mlp_actor_critic`** unless noted. Observation and action shapes follow the YAML `env` / `training_mode` setup (flattened branched actions where `flatten_branched: true`).

| Package | Representative config(s) | Training summary |
|--------|---------------------------|------------------|
| [`single_ppo_checkpoint_agent/`](single_ppo_checkpoint_agent/) | e.g. [`configs/train_single_ppo_still.yaml`](../configs/train_single_ppo_still.yaml) or any single-agent PPO config with per-player control | **Single-agent** PPO: 336-dim vector per player, **Discrete(27)** flattened actions, typically **hidden [256, 256]** (match `model.hidden_sizes` in your run to `HIDDEN_SIZES` in `agent.py`). |
| [`single_ppo_dense_ceia_checkpoint_agent/`](single_ppo_dense_ceia_checkpoint_agent/) | [`configs/train_single_ppo_dense_vs_ceia_self_teammate.yaml`](../configs/train_single_ppo_dense_vs_ceia_self_teammate.yaml) | **vs CEIA baseline** opponent, **self-teammate** (shared policy), **dense reward shaping** + sparse goals, **hidden [256, 256]**. |
| [`ppo_dense_agent_ceia_trained/`](ppo_dense_agent_ceia_trained/) | [`configs/train_single_ppo_dense_self_play_10m.yaml`](../configs/train_single_ppo_dense_self_play_10m.yaml) (and related dense self-play runs) | Long-run **dense** single-agent PPO with **self-play opponent** options; **hidden [512, 512]** to match the policy head in this folder. A follow-up run that freezes this checkpoint as the opponent is [`configs/train_single_ppo_dense_vs_ppo_dense_agent_ceia_trained.yaml`](../configs/train_single_ppo_dense_vs_ppo_dense_agent_ceia_trained.yaml). |
| [`team_ppo_dense_agent/`](team_ppo_dense_agent/) | [`configs/train_team_ppo_dense_vs_random.yaml`](../configs/train_team_ppo_dense_vs_random.yaml) | **Team** PPO (`single_agent: false`): **672**-dim observation (two players concatenated), **729** joint discrete actions (27×27), **vs random** opponent team, **dense rewards**, **hidden [256, 256]**. |

If you change **hidden sizes** or **architecture** in training, update the constants at the top of that agent’s `agent.py` (and ensure `model.py` stays consistent).

## Checkpoints

After training, copy your best checkpoint (e.g. from `runs/<run>/checkpoints/`) into the agent folder as `checkpoint.pth` (or set the env var documented in that agent’s `README.md`).

For quick mapping without reading the table:

- **Team PPO** (joint 729 actions, 672-dim concat obs): [`team_ppo_dense_agent/`](team_ppo_dense_agent/)
- **Single-player PPO** (per-player 27 actions, 336-dim obs): [`single_ppo_checkpoint_agent/`](single_ppo_checkpoint_agent/)
- **Single-player dense vs CEIA + self-teammate** (256×256 head): [`single_ppo_dense_ceia_checkpoint_agent/`](single_ppo_dense_ceia_checkpoint_agent/)
- **Dense self-play / larger 512×512 head** (this repo’s “CEIA trained” export): [`ppo_dense_agent_ceia_trained/`](ppo_dense_agent_ceia_trained/)
