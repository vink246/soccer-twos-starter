"""Named policies for teammates / opponents (string presets in YAML)."""

from typing import Callable

import gym


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

    raise ValueError(f"Unknown policy preset: {name!r} (known: still, random)")


def build_team_opponent_policy(name: str, action_space: gym.spaces.Discrete) -> Callable:
    """Policy for the opposing team in multiagent_team rollouts (obs -> int action)."""
    return build_policy(name, action_space)
