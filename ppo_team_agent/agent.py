"""PPO team policy (``team_shared``) for soccer_twos.watch."""

from soccer_rl.common.ppo_submission_agent import PPOSubmissionAgent


class PPOTeamAgent(PPOSubmissionAgent):
    """
    One shared policy per team (RLlib policy ids ``team_0`` / ``team_1``).

    Train with ``multiagent.policy_mode: team_shared`` in the same YAML you pass
    as ``training_config_path``.
    """

    display_name = "PPO Team (shared policy)"
    expected_policy_mode = "team_shared"
