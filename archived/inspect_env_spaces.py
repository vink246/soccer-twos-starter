"""
Print observation / action / reward structure for Soccer-Twos variants.

Run: conda run -n soccertwos python inspect_env_spaces.py

Requires the Unity soccer-twos binary (installed with the soccer-twos package).
"""
import numpy as np
import gym
import soccer_twos
from soccer_twos import EnvType


def describe_space(name, space):
    print(f"  {name}: {type(space).__name__} = {space}")
    if hasattr(space, "shape"):
        print(f"    shape: {space.shape}")
    if hasattr(space, "n"):
        print(f"    n: {space.n}")
    if hasattr(space, "nvec") and space.nvec is not None:
        print(f"    nvec: {space.nvec}")


def summarize_array(arr, label="arr"):
    a = np.asarray(arr)
    return f"{label} shape={a.shape} dtype={a.dtype} min={a.min():.4g} max={a.max():.4g}"


def print_first_step(tag, obs, reward, done, info):
    print(f"\n  --- First step after reset ({tag}) ---")
    if isinstance(obs, dict):
        print("  obs: dict keys", sorted(obs.keys()))
        for k in sorted(obs.keys()):
            print("   ", k, summarize_array(obs[k], "obs"))
    else:
        print(" ", summarize_array(obs, "obs"))
    print("  reward:", reward, "| type:", type(reward))
    print("  done:", done, "| type:", type(done))
    if isinstance(info, dict) and info:
        k0 = next(iter(info))
        inner = info[k0]
        print("  info: keys", sorted(info.keys()))
        if isinstance(inner, dict):
            print("    info[%r] keys: %s" % (k0, list(inner.keys())))
            if "player_info" in inner:
                print("    sample player_info:", inner["player_info"])
            if "ball_info" in inner:
                print("    sample ball_info:", inner["ball_info"])
        else:
            print("    info[%r]:" % k0, inner)


def rollout(env, step_fn, n_steps=25, tag=""):
    rewards_flat = []
    dones = 0
    for t in range(n_steps):
        obs, reward, done, info = step_fn()
        if t == 0:
            print_first_step(tag, obs, reward, done, info)
        if isinstance(reward, dict):
            rewards_flat.extend(float(r) for r in reward.values())
            d = done["__all__"] if isinstance(done, dict) and "__all__" in done else any(done.values())
        else:
            rewards_flat.append(float(reward))
            d = bool(done)
        if d:
            dones += 1
            env.reset()
    r = np.array(rewards_flat, dtype=np.float64)
    print(f"\n  --- Summary over {n_steps} steps ({tag}) ---")
    print(f"  per-decision rewards: min={r.min():.4g} max={r.max():.4g} mean={r.mean():.4g}")
    print(f"  nonzero reward decisions: {(r != 0).sum()} / {len(r)}")
    print(f"  episode ends (done) encountered: {dones}")


def main():
    # Distinct base ports in case a prior run left a process
    cfgs = [
        ("single_player", 5010),
        ("multiagent_player", 5011),
        ("multiagent_team", 5012),
    ]

    common = dict(render=False, flatten_branched=True, opponent_policy=lambda *_: 0)

    for name, port in cfgs:
        print("\n" + "=" * 72)
        if name == "single_player":
            print("A) Single controlled player: team_vs_policy, single_player=True")
            env = soccer_twos.make(
                **common,
                base_port=port,
                variation=EnvType.team_vs_policy,
                single_player=True,
            )
        elif name == "multiagent_player":
            print("B) Multi-agent (per player): multiagent_player")
            env = soccer_twos.make(
                **common,
                base_port=port,
                variation=EnvType.multiagent_player,
            )
        else:
            print("C) Multi-agent (per team): multiagent_team")
            env = soccer_twos.make(
                **common,
                base_port=port,
                variation=EnvType.multiagent_team,
            )

        try:
            describe_space("observation_space", env.observation_space)
            describe_space("action_space", env.action_space)

            o0 = env.reset()
            print("\n  reset() return:")
            if isinstance(o0, dict):
                print("   dict keys:", sorted(o0.keys()))
                for k in sorted(o0.keys()):
                    print("   ", k, summarize_array(o0[k], "obs"))
            else:
                print(" ", summarize_array(o0, "obs"))

            if name == "single_player":

                def step_fn():
                    return env.step(env.action_space.sample())

            elif name == "multiagent_player":

                def step_fn():
                    act = {i: env.action_space.sample() for i in range(4)}
                    return env.step(act)

            else:

                def step_fn():
                    # Two teams: keys 0 and 1
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        return env.step(
                            np.array([env.action_space.sample(), env.action_space.sample()])
                        )
                    return env.step(
                        {
                            0: env.action_space[0].sample(),
                            1: env.action_space[1].sample(),
                        }
                    )

            rollout(env, step_fn, n_steps=25, tag=name)
        finally:
            env.close()

    print("\n" + "=" * 72)
    print("Reward timing (from soccer_twos wrappers + ML-Agents):")
    print("  Each env.step() returns the reward Unity emitted for that decision step.")
    print("  Wrapper uses: reward[i] + group_reward[i] per agent (see MultiAgentUnityWrapper).")
    print("  When values are nonzero (goals, etc.) is defined in the Unity C# environment,")
    print("  not in this Python repo — inspect soccer-twos-env or log nonzero steps above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
