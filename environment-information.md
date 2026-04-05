# Soccer-Twos environment (reference)

Living notes for this starter repo: what the Python / Unity stack exposes. **Add new bullets or sections here** whenever we discover more (e.g. from [soccer-twos-env](https://github.com/bryanoliveira/soccer-twos-env), ML-Agents, or experiments).

**Package:** `soccer-twos` (pip) wraps a Unity binary and uses `mlagents_envs`. Wrapper source lives in `site-packages/soccer_twos/wrappers.py` (not this repo).

---

## Core vector observation

- Each **player** gets a **336-dimensional** `float32` vector (normalized ray / sensor style values in practice; exact semantics are defined in Unity).
- Unity may send **345** floats per agent; the wrapper keeps the first **336** for learning and, when present, puts the rest into `info` as `player_info` and `ball_info` (see below).

---

## Variants (EnvType)

### A. Single controlled player — `EnvType.team_vs_policy` + `single_player=True`

| Aspect | Shape / type |
|--------|----------------|
| **Observation** | `np.ndarray` shape **(336,)**, `float32`; Gym declares `Box` with low/high in **[0, 1]**. |
| **Action** | **`Discrete(27)`** when `flatten_branched=True` (default in many examples): one integer per step. |
| **`reset` / `step` → `obs`** | Single array, not a dict. |
| **Reward** | **Scalar** `float32`: sum of team 0’s two players (`reward[0] + reward[1]`) in the underlying 4-player env. |
| **`done`** | **`bool`**. |
| **`info`** | **Flat** `dict`: keys **`player_info`**, **`ball_info`** (for the controlled perspective / stripped tail of the obs). |

Opponents use `opponent_policy`; teammate can use `teammate_policy` (default stays still).

### B. Multi-agent per player — `EnvType.multiagent_player`

| Aspect | Shape / type |
|--------|----------------|
| **Observation** | **`dict`** `{0, 1, 2, 3} →` array **(336,)** each. |
| **Action** | **`dict`** `{0: a0, 1: a1, 2: a2, 3: a3}`** with each `a*` in **`Discrete(27)`** when flattened. |
| **`env.observation_space`** | Still a single shared **`Box(336,)`** on the env object; runtime API is always the dict of four agents. |
| **Reward** | **`dict`** per agent id → scalar. |
| **`done`** | **`{'__all__': bool}`** — episode-level flag (termination mode configurable via `termination_mode` in `make()`). |
| **`info`** | **`dict`** keyed by agent id; each value has **`player_info`** and **`ball_info`**. |

### C. Multi-agent per team — `EnvType.multiagent_team`

| Aspect | Shape / type |
|--------|----------------|
| **Observation** | **`dict`** `{0, 1}` (two teams). Each value is **(672,)** = **concat(obs_player_A, obs_player_B)** on that team (2 × 336). |
| **Action** | **`Discrete(729)`** = **27²** per team: one joint index encodes both players’ `Discrete(27)` actions. Pass e.g. **`np.array([a_team0, a_team1])`** (two integers). |
| **Reward** | **`dict`** `{0: r0, 1: r1}`** — each is **sum of the two players’** rewards on that team. |
| **`done`** | **`{'__all__': bool}`** (same pattern as per-player multi-agent). |
| **`info`** | **Nested:** `info[team_id][player_id]` → per-player dicts (e.g. `player_info`, `ball_info`). |

---

## Reward mechanics (Python side)

- Every **`env.step(...)`** returns rewards **for that ML-Agents decision step**.
- Under the hood: **`reward[i] + group_reward[i]`** per agent before team/single-player aggregation (`MultiAgentUnityWrapper._single_step`).
- **When** rewards are non-zero (goals, shaping, timeouts) is defined in the **Unity C#** project, not in this starter — see **soccer-twos-env** for authoritative behavior.
- Short random rollouts often see **all zeros** until a goal or episode boundary; that is expected.

---

## Dense reward shaping (PyTorch training)

This repo can add **weighted dense terms** on top of Unity’s sparse team reward when you use `scripts/train.py` (or any code path that builds the env via `soccer_rl.env_factory.make_env` / `make_env_from_flat_config`).

**YAML (top level, next to `env:`):**

```yaml
dense_reward:
  enabled: true
  sparse_weight: 1.0          # multiply Unity reward before adding dense terms
  clip: 0.5                   # optional: abs clip on total dense per step
  terms:
    ball_vel_attack_component:
      weight: 0.02
      axis: 0                  # 0 = x, 1 = y in world (x,y) from info
      attack_sign: 1.0         # flip sign if your pitch direction is reversed
    screen_own_goal:
      weight: 0.05
      own_goal_xy: [-1.0, 0.0]
      opponent_goal_xy: [1.0, 0.0]
    hide_ball_from_opponent_los:
      weight: 0.03
      opponent_pos_obs: [120, 121]   # indices into the 336-dim vector (you must calibrate)
      opponent_yaw_obs: 122
      yaw_in_degrees: true
      fov_degrees: 120
      max_range: 50.0
```

**Requirements:** Most geometric terms use **`player_info` / `ball_info`** in `info`, which appear only when the binary sends **345** floats per agent (wrapper strips the last 9 into `info` and keeps **336** for the policy). If your build only sends 336, those terms are **0** until you enable the extra channels in Unity or add new terms that read from fixed observation indices.

**Implementation:** `soccer_rl/dense_rewards.py` registers term names; `soccer_rl/dense_reward_wrapper.py` wraps the Gym env. See **`configs/train_single_ppo_dense_example.yaml`** for a commented template.

**List of built-in term names:** `ball_attack_axis_delta`, `ball_vel_attack_component`, `ball_own_goal_threat_gaussian`, `distance_to_ball_closer_delta`, `screen_own_goal`, `ball_opponent_goal_potential_delta`, `ball_distance_to_own_goal`, `ball_distance_to_opponent_goal`, `ball_own_times_opp_goal_distance`, `hide_ball_from_opponent_los`.

---

## Networking / ports (ML-Agents)

- GRPC uses **`port = base_port + worker_id`** (defaults in `make`: `base_port=50039`, `worker_id=0`).
- **`UnityWorkerInUseException` / address already in use:** another process holds that port, or a previous env did not shut down — change **`worker_id`** or **`base_port`**, or close stray **`soccer-twos`** / Unity processes.

---

## `info` extras (when obs length is 345)

From wrapper logic (binary-dependent):

- **`player_info`:** `position` (2,), `rotation_y` (scalar), `velocity` (2,)
- **`ball_info`:** `position` (2,), `velocity` (2,)

---

## Document history

| Date | Notes |
|------|--------|
| 2026-04-01 | Initial write-up from `inspect_env_spaces.py` output and `soccer_twos.wrappers` (single-player, multiagent_player, multiagent_team; rewards; ports; info schema). |
| 2026-04-02 | Documented optional `dense_reward` YAML block and `soccer_rl` dense shaping wrapper for PyTorch training. |
