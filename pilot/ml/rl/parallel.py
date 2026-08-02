"""Multiprocessing trainer: collect experience from many games at once.

The single-process trainer steps one game at a time on effectively one core, and
Python threads can't fix it — the per-step work (network math, JSON) is GIL-held,
so `train_parallel` (threads) only reached ~1.3x. Separate PROCESSES escape the
GIL: each worker runs its own game on its own core.

Design (synchronous rounds, simplest thing that's correct):
  1. the learner broadcasts its current weights,
  2. N worker processes each play `episodes_per_round` games in parallel with
     those weights (a persistent env + a local act-only policy per worker),
  3. they ship the transitions back, the learner adds them to its replay buffer
     and takes gradient steps, then round repeats.

Workers act on the round's (slightly stale) weights — standard in distributed RL.
Only the learner trains, so there is one authoritative network.
"""

from __future__ import annotations

import multiprocessing as mp
from statistics import mean
from typing import Any, Callable, Optional

import numpy as np

from .env import GameEnv
from .reward import RewardSpec

# --- per-worker globals (each worker process holds a persistent env + policy) ---
_ENV: Optional[GameEnv] = None
_AGENT = None
_FEAT: Optional[Callable] = None
_REWARD: Optional[RewardSpec] = None
_RNG: Optional[np.random.Generator] = None
_ACTIONS: list = []


def _worker_init(env_factory, env_kwargs, base_seed, agent_params, featurizer, reward):
    """Runs once per worker process: build the env (its own game process) and a
    local act-only agent that will be refreshed with the learner's weights. Each
    worker gets a DISTINCT seed from its pool identity, so they play different
    games (identical seeds would just collect the same experience N times)."""
    global _ENV, _AGENT, _FEAT, _REWARD, _RNG, _ACTIONS
    from .dqn import DQNAgent
    ident = mp.current_process()._identity
    seed = base_seed + 1000 * (ident[0] if ident else 0)
    _ENV = env_factory(seed, **env_kwargs)
    _AGENT = DQNAgent(**agent_params)
    _AGENT._freeze_scale = True            # act on the learner's synced scale, read-only
    _FEAT = featurizer
    _REWARD = reward
    _RNG = np.random.default_rng(seed)
    _ACTIONS = list(agent_params["actions"])


def _worker_collect(weights, n_episodes, epsilon):
    """Play `n_episodes` with the given weights; return (transitions, returns).
    Transitions are already featurized vectors, ready for the learner's buffer."""
    _AGENT.import_weights(weights)
    _AGENT._freeze_scale = True
    have_net = _AGENT._net is not None
    transitions, returns = [], []
    for _ in range(n_episodes):
        obs = _ENV.reset()
        vec = _AGENT._vec(_FEAT(obs))
        done, total = False, 0.0
        while not done:
            if not have_net or _RNG.random() < epsilon:
                a_idx = int(_RNG.integers(len(_ACTIONS)))
            else:
                q, _ = _AGENT._net.forward(vec[None, :])
                a_idx = int(np.argmax(q[0]))
            nxt, done, info = _ENV.step(_ACTIONS[a_idx])
            nvec = _AGENT._vec(_FEAT(nxt))
            r = _REWARD.compute(obs, nxt, done, info)
            transitions.append((vec, a_idx, float(r), nvec, bool(done)))
            total += r
            obs, vec = nxt, nvec
        returns.append(total)
    return transitions, returns


def train_parallel_mp(
    agent, env_factory: Callable, env_kwargs: dict, agent_params: dict,
    featurizer: Callable, reward: RewardSpec, *,
    n_workers: int, episodes: int, episodes_per_round: int = 4,
    epsilon_start: float = 1.0, epsilon_final: float = 0.05,
    probe_env: Optional[GameEnv] = None,
) -> list[float]:
    """Train `agent` from `n_workers` parallel games. Returns the learning curve
    (mean episode return per ~5% of episodes), same shape as `train`.

    `probe_env` (main-process env) is used once to build the learner's network and
    feature schema before the first broadcast, so workers get real weights from
    round 1. If None, one is built from env_factory.
    """
    # build the learner's net + feature schema up front (it never featurizes
    # during training — it only ingests worker-featurized vectors)
    own_probe = probe_env is None
    if own_probe:
        probe_env = env_factory(0, **env_kwargs)
    agent._vec(featurizer(probe_env.reset()))    # locks keys, builds the net
    if own_probe and hasattr(probe_env, "close"):
        probe_env.close()

    span = max(epsilon_start - epsilon_final, 0.0)
    per_round_total = n_workers * episodes_per_round
    block = max(1, episodes // 20)
    curve: list[float] = []
    window: list[float] = []
    done_eps = 0

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        n_workers, initializer=_worker_init,
        initargs=(env_factory, env_kwargs, 1, agent_params, featurizer, reward),
    )
    try:
        while done_eps < episodes:
            eps = max(epsilon_final, epsilon_start - span * done_eps / max(1, episodes))
            weights = agent.export_weights()
            results = pool.starmap(
                _worker_collect, [(weights, episodes_per_round, eps)] * n_workers)
            for transitions, returns in results:
                for vec, a_idx, r, nvec, done in transitions:
                    agent.store_vec(vec, a_idx, r, nvec, done)
                    agent._steps += 1
                    if agent._steps % agent.learn_every == 0:
                        agent.train_step()
                for ret in returns:
                    window.append(ret)
                    if len(window) >= block:
                        curve.append(round(mean(window), 2))
                        window.clear()
            done_eps += per_round_total
    finally:
        pool.close()
        pool.join()
    if window:
        curve.append(round(mean(window), 2))
    return curve
