"""PPO per-player policies for soccer_twos.watch."""

from soccer_rl.common.ppo_submission_agent import PPOSubmissionAgent


class PPOPlayerAgent(PPOSubmissionAgent):
    """
    Separate policy per player (RLlib policy ids ``player_0`` … ``player_3``).

    Train with ``multiagent.policy_mode: per_player`` using the same YAML you pass
    as ``training_config_path``.
    """

    display_name = "PPO (per-player policies)"
    expected_policy_mode = "per_player"
