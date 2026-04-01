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
