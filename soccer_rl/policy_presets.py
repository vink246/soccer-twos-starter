"""Named policies for teammates / opponents (string presets in YAML)."""

from typing import Any, Callable, Dict, Optional

import gym
import numpy as np


def _rllib_action_to_flat27(action: Any) -> int:
    """
    RLlib ``compute_single_action`` may return a numpy scalar, shape-(1,) array, or a
    MultiDiscrete branch vector (Soccer-Twos typically [3,3,3] → flat 0..26).
    """
    arr = np.asarray(action, dtype=np.int64).ravel()
    if arr.size == 0:
        raise ValueError("empty action from RLlib policy")
    if arr.size == 1:
        return int(arr.flat[0])
    branch_sizes = (3, 3, 3)
    if arr.size != len(branch_sizes):
        raise ValueError(
            "CEIA action has unexpected size %d (expected 1 or %d): %r"
            % (arr.size, len(branch_sizes), action)
        )
    idx = 0
    mul = 1
    for i in range(len(branch_sizes) - 1, -1, -1):
        idx += int(arr[i]) * mul
        mul *= branch_sizes[i]
    return idx


def _build_ceia_baseline_team_policy(action_space: gym.spaces.Discrete) -> Callable[[Any], int]:
    """
    Opponent = CEIA Ray RLlib policy (per-player), composed into joint Discrete(729).

    Requires ``agents/ceia_baseline_agent`` with checkpoint + ``params.pkl`` as in the
    baseline bundle (same as ``RayAgent``).
    """
    if action_space.n != 729:
        raise ValueError(
            "ceia_baseline opponent expects joint team Discrete(729); got n=%d" % action_space.n
        )

    import gym as gym_lib

    from agents.ceia_baseline_agent.agent_ray import RayAgent

    class _Shim(gym_lib.Env):
        """RayAgent requires a gym.Env; policy inference ignores it."""

        metadata = {}

        def __init__(self):
            super().__init__()
            self.observation_space = gym_lib.spaces.Box(
                -np.inf, np.inf, shape=(336,), dtype=np.float32
            )
            self.action_space = gym_lib.spaces.Discrete(27)

        def reset(self):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

    holder: Dict[str, Optional[RayAgent]] = {"agent": None}

    def policy(obs_team: Any) -> int:
        if holder["agent"] is None:
            holder["agent"] = RayAgent(_Shim())
        agent = holder["agent"]
        o = np.asarray(obs_team, dtype=np.float32).reshape(-1)
        if o.size != 672:
            raise ValueError(
                "CEIA team opponent expected 672-dim observation; got %d" % o.size
            )
        a0, *_ = agent.policy.compute_single_action(o[:336])
        a1, *_ = agent.policy.compute_single_action(o[336:])
        return _rllib_action_to_flat27(a0) * 27 + _rllib_action_to_flat27(a1)

    return policy


def _build_ceia_baseline_player_policy(action_space: gym.spaces.Discrete) -> Callable[[Any], int]:
    """
    Opponent = CEIA Ray RLlib policy for single-player/team_vs_policy rollouts.

    Expects per-player 336-dim observations, returns flat Discrete(27) action indices.
    """
    if action_space.n != 27:
        raise ValueError(
            "ceia_baseline player policy expects Discrete(27); got n=%d" % action_space.n
        )

    import gym as gym_lib

    from agents.ceia_baseline_agent.agent_ray import RayAgent

    class _Shim(gym_lib.Env):
        metadata = {}

        def __init__(self):
            super().__init__()
            self.observation_space = gym_lib.spaces.Box(
                -np.inf, np.inf, shape=(336,), dtype=np.float32
            )
            self.action_space = gym_lib.spaces.Discrete(27)

        def reset(self):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

    holder: Dict[str, Optional[RayAgent]] = {"agent": None}

    def policy(obs_player: Any) -> int:
        if holder["agent"] is None:
            holder["agent"] = RayAgent(_Shim())
        agent = holder["agent"]
        o = np.asarray(obs_player, dtype=np.float32).reshape(-1)
        if o.size != 336:
            raise ValueError(
                "CEIA player opponent expected 336-dim observation; got %d" % o.size
            )
        a, *_ = agent.policy.compute_single_action(o)
        return _rllib_action_to_flat27(a)

    return policy


def build_policy(name: str, action_space: gym.Space) -> Callable:
    """
    Return a callable suitable for soccer_twos opponent_policy / teammate_policy
    (single-agent team_vs_policy) or for use in trainers (multiagent_team).

    For Discrete(flattened) spaces, callables return an int action index.
    """
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"Expected gym.spaces.Discrete, got {type(action_space)}")

    n = action_space.n
    key = (name or "random").lower().replace("-", "_")

    if key in ("still", "do_nothing", "zero"):
        return lambda *_args, **_kwargs: 0

    if key == "random":
        return lambda *_args, **_kwargs: int(action_space.sample())

    if key in ("uniform_random",):
        return lambda *_args, **_kwargs: int(action_space.sample())

    if key in ("ceia_baseline", "ceia"):
        if n == 729:
            return _build_ceia_baseline_team_policy(action_space)
        if n == 27:
            return _build_ceia_baseline_player_policy(action_space)
        raise ValueError(
            "ceia_baseline preset requires Discrete(27) or Discrete(729), got n=%d" % n
        )

    raise ValueError(
        f"Unknown policy preset: {name!r} (known: still, random, ceia_baseline)"
    )


def build_team_opponent_policy(name: str, action_space: gym.spaces.Discrete) -> Callable:
    """Policy for the opposing team in multiagent_team rollouts (obs -> int action)."""
    return build_policy(name, action_space)
