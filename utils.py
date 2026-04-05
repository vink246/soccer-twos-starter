from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv

from soccer_rl.env_factory import make_env_from_flat_config, normalize_worker_env_config


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable or preset string (still, random) for your agent to train against.
    """
    env = make_env_from_flat_config(env_config)
    # env = TransitionRecorderWrapper(env)
    cfg = normalize_worker_env_config(env_config)
    if "multiagent" in cfg and not cfg["multiagent"]:
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
