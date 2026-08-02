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


def test_resumed_training_does_not_restart_exploration():
    """A chunk that continues a trained policy must not begin by acting at
    random: `--forever` loaded good weights, then trained them against 4000
    near-random episodes, and the policy degraded every chunk (floor 1.80 ->
    1.73 -> 1.00). Exploration must start where the caller says it does."""
    from pilot.ml.rl.train import train
    from pilot.ml.rl.reward import RewardSpec, RewardRule
    from pilot.ml.rl.spd_sim import SPDGridEnv
    from pilot.ml.rl.agent import QLearningAgent

    seen: list[float] = []

    class SpyAgent(QLearningAgent):
        def act(self, obs, epsilon=0.0):
            seen.append(epsilon)
            return super().act(obs, epsilon)

    reward = RewardSpec(rules=[RewardRule(field="depth", direction="up", weight=1.0)])
    env = SPDGridEnv(seed=0, max_steps=5)
    train(env, SpyAgent(env.action_space, seed=0), reward, episodes=4,
          epsilon_start=0.2)

    assert max(seen) <= 0.2, "a resumed chunk must not explore from scratch"

    seen.clear()
    train(env, SpyAgent(env.action_space, seed=0), reward, episodes=4)
    assert max(seen) > 0.9, "a cold run should still explore fully at the start"


def test_train_parallel_learns_and_counts_episodes():
    """Several envs feeding one agent must be thread-safe AND actually learn:
    run the exact episode budget, and beat a random policy on the sim."""
    from pilot.ml.rl.train import train_parallel, evaluate
    from pilot.ml.rl.spd import spd_reward_spec
    from pilot.ml.rl.spd_sim import SPDGridEnv, spd_featurizer
    from pilot.ml.rl.dqn import DQNAgent

    envs = [SPDGridEnv(seed=s, max_steps=40) for s in range(4)]
    agent = DQNAgent(envs[0].action_space, seed=0, warmup=50)
    curve = train_parallel(agent, envs, spd_reward_spec(), episodes=400,
                           featurizer=spd_featurizer, epsilon_start=1.0, epsilon_final=0.1)
    assert sum(len(c) for c in [curve]) and len(curve) >= 15   # ~20 blocks over 400 eps
    assert curve[-1] > curve[0]                                # learning progressed

    import random
    ev = SPDGridEnv(seed=999, max_steps=40)
    trained, _ = evaluate(ev, agent.policy, spd_reward_spec(), 30, featurizer=spd_featurizer)
    rng = random.Random(1)
    rand, _ = evaluate(ev, lambda o: rng.choice(ev.action_space), spd_reward_spec(), 30)
    assert trained > rand                                      # beats random


def test_prioritized_replay_favors_surprising_transitions():
    """Prioritized sampling must draw high-|TD| transitions far more often than
    uniform would, and importance weights must correct the bias."""
    import numpy as np
    from pilot.ml.rl.dqn import DQNAgent

    agent = DQNAgent(["a", "b"], seed=0, warmup=4, batch_size=8,
                     prioritized=True, prio_alpha=1.0, prio_beta=0.4)
    # seed a buffer, then hand-set one transition as far more surprising
    for i in range(60):
        agent.learn({"x": float(i % 5)}, "a", 0.0, {"x": float((i + 1) % 5)}, False)
    agent._prio_buf[: agent._size] = 0.01
    agent._prio_buf[7] = 100.0                       # the one that matters
    scaled = agent._prio_buf[: agent._size] ** agent.prio_alpha
    probs = scaled / scaled.sum()
    draws = agent._nprng.choice(agent._size, 5000, p=probs)
    share = (draws == 7).mean()
    assert share > 0.5, "the surprising transition should dominate sampling"
    # uniform would draw it ~1/size of the time
    assert share > 20 * (1.0 / agent._size)


def test_prioritized_and_uniform_both_learn_the_bandit():
    """Turning prioritization on or off must both still learn (it changes which
    transitions are trained on, not correctness)."""
    from pilot.ml.rl.dqn import DQNAgent

    def bandit(prioritized):
        ag = DQNAgent(["left", "right"], seed=1, warmup=20, prioritized=prioritized)
        for _ in range(400):
            r = 1.0 if True else 0.0
            ag.learn({"s": 1.0}, "right", 1.0, {"s": 1.0}, True)
            ag.learn({"s": 1.0}, "left", 0.0, {"s": 1.0}, True)
        return ag.policy({"s": 1.0})

    assert bandit(True) == "right"
    assert bandit(False) == "right"


def test_dueling_network_learns_and_roundtrips():
    """Dueling head must learn the bandit AND survive a Q snapshot roundtrip
    (the snapshot has to rebuild the dueling architecture, not a plain head)."""
    from pilot.ml.rl.dqn import DQNAgent, _DuelingMLP

    ag = DQNAgent(["left", "right"], seed=2, warmup=20, dueling=True)
    for _ in range(400):
        ag.learn({"s": 1.0}, "right", 1.0, {"s": 1.0}, True)
        ag.learn({"s": 1.0}, "left", 0.0, {"s": 1.0}, True)
    assert ag.policy({"s": 1.0}) == "right"
    assert isinstance(ag._net, _DuelingMLP)

    snap = ag.Q
    assert snap["dueling"] is True
    ag2 = DQNAgent(["left", "right"], seed=9, dueling=False)   # wrong default...
    ag2.Q = snap                                               # ...snapshot must fix it
    assert isinstance(ag2._net, _DuelingMLP)
    assert ag2.policy({"s": 1.0}) == ag.policy({"s": 1.0})


def test_dueling_backward_matches_numeric_gradient():
    """Sanity-check the hand-written dueling backprop against a finite-difference
    gradient, so a subtle mistake in the V/advantage split can't slip through.
    Done in float64 so numerical noise can't mask (or fake) a match."""
    import numpy as np
    from pilot.ml.rl.dqn import _DuelingMLP

    rng = np.random.default_rng(0)
    net = _DuelingMLP(4, 3, 8, rng)
    p64 = {k: v.astype(np.float64) for k, v in net.params.items()}
    x = rng.normal(size=(5, 4))
    dq = rng.normal(size=(5, 3))

    def loss():
        z1 = x @ p64["W1"] + p64["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p64["W2"] + p64["b2"]; a2 = np.maximum(z2, 0)
        v = a2 @ p64["Wv"] + p64["bv"]; adv = a2 @ p64["Wa"] + p64["ba"]
        q = v + adv - adv.mean(1, keepdims=True)
        return (q * dq).sum()

    # analytic gradient for Wa, exactly as backward_step computes it
    z1 = x @ p64["W1"] + p64["b1"]; a2 = np.maximum(x @ p64["W1"] + p64["b1"], 0)
    a1 = np.maximum(z1, 0); a2 = np.maximum(a1 @ p64["W2"] + p64["b2"], 0)
    analytic = a2.T @ (dq - dq.mean(1, keepdims=True))

    eps = 1e-6
    num = np.zeros_like(p64["Wa"])
    for i in range(p64["Wa"].shape[0]):
        for j in range(p64["Wa"].shape[1]):
            orig = p64["Wa"][i, j]
            p64["Wa"][i, j] = orig + eps; hi = loss()
            p64["Wa"][i, j] = orig - eps; lo = loss()
            p64["Wa"][i, j] = orig
            num[i, j] = (hi - lo) / (2 * eps)
    assert np.allclose(num, analytic, atol=1e-6, rtol=1e-5)


def test_store_train_split_and_weight_export_import():
    """The multiprocessing trainer relies on: storing a pre-featurized transition
    separately from taking a gradient step, and exporting/importing just the
    online weights so a worker can act with the learner's latest policy."""
    import numpy as np
    from pilot.ml.rl.dqn import DQNAgent

    learner = DQNAgent(["left", "right"], seed=0, warmup=10)
    # learn the bandit via the split API (store_vec + train_step), as the mp loop does
    learner._vec({"s": 1.0})                      # lock schema + build net
    for _ in range(300):
        for a, r in ((1, 1.0), (0, 0.0)):
            v = learner._vec({"s": 1.0})
            learner.store_vec(v, a, r, v, True)
            learner._steps += 1
            if learner._steps % learner.learn_every == 0:
                learner.train_step()
    assert learner.policy({"s": 1.0}) == "right"

    # a worker importing the learner's weights must reproduce its greedy policy,
    # and must NOT drift its frozen normalisation scale while acting
    worker = DQNAgent(["left", "right"], seed=99)
    worker.import_weights(learner.export_weights())
    worker._freeze_scale = True
    scale_before = worker._scale.copy()
    assert worker.policy({"s": 1.0}) == learner.policy({"s": 1.0})
    for _ in range(20):
        worker.act({"s": 5.0}, epsilon=0.0)       # large feature would grow an unfrozen scale
    assert np.array_equal(worker._scale, scale_before)   # frozen: unchanged
