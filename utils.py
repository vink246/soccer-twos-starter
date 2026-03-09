from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

# Re-export for backward compatibility (example_ray_*.py, train_ray_*.py).
from training_utils import create_rllib_env


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """RLlib wrapper so env can inherit from MultiAgentEnv. Prefer training_utils.create_rllib_env for training."""
    pass


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
