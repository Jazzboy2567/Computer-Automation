"""DQN agent: interface parity, learning on a trivial task, snapshot roundtrip."""

from __future__ import annotations

from pilot.ml.rl.dqn import DQNAgent
from pilot.ml.rl.env import SimEnv
from pilot.ml.rl.reward import RewardSpec
from pilot.ml.rl.train import train


def test_learns_a_contextual_bandit():
    """Feature tells which action pays; the net must learn the mapping."""
    agent = DQNAgent(["left", "right"], seed=0, warmup=64, learn_every=1, sync_every=50)
    import random
    rng = random.Random(1)
    for _ in range(3000):
        ctx = rng.choice([0.0, 1.0])
        obs = {"ctx": ctx, "noise": rng.random()}
        action = agent.act(obs, epsilon=0.3)
        correct = (action == "right") == (ctx == 1.0)   # ctx 1 -> right, ctx 0 -> left
        agent.learn(obs, action, 1.0 if correct else -1.0, obs, True)

    hits = sum(agent.policy({"ctx": c, "noise": 0.5}) == ("right" if c else "left")
               for c in (0.0, 1.0, 0.0, 1.0))
    assert hits == 4, "net failed to learn a 2-context bandit"


def test_trains_on_sim_env_via_generic_loop():
    """Drop-in compatibility with train() (identity featurizer, full obs)."""
    env = SimEnv(seed=3)
    agent = DQNAgent(env.action_space, seed=0, warmup=32, learn_every=2)
    curve = train(env, agent, RewardSpec.survival_default(), episodes=30)
    assert len(curve) >= 1 and agent.states_learned > 0


def test_snapshot_roundtrip():
    agent = DQNAgent(["a", "b"], seed=0, warmup=8, learn_every=1)
    for i in range(64):
        obs = {"x": float(i % 2)}
        agent.learn(obs, "a" if i % 2 else "b", 1.0, obs, True)
    snap = agent.Q
    assert snap and "params" in snap

    clone = DQNAgent(["a", "b"], seed=1)
    clone.Q = snap
    probe = {"x": 1.0}
    assert clone.policy(probe) == agent.policy(probe)


def test_unseen_keys_default_to_zero():
    agent = DQNAgent(["a", "b"], seed=0)
    agent.policy({"x": 1.0, "y": 2.0})            # locks the schema
    assert agent.policy({"x": 1.0}) in ("a", "b")  # missing key -> 0, no crash
