"""Training loop + evaluation for the RL agent.

Reward is computed from observation changes by the `RewardSpec`, not the env, so
the same loop drives the simulated game and (later) the real one.
"""

from __future__ import annotations

import threading
from statistics import mean
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .agent import QLearningAgent
from .env import GameEnv, Observation
from .reward import RewardSpec

Policy = Callable[[Observation], str]
Featurizer = Callable[[Observation], Observation]


def _identity(obs: Observation) -> Observation:
    return obs


class RLResult(BaseModel):
    """How well the trained agent performs vs a random baseline."""

    episodes: int
    actions: list[str]
    avg_return_trained: float
    avg_return_random: float
    avg_survival_trained: float
    avg_survival_random: float
    improvement: float
    states_learned: int
    learning_curve: list[float] = Field(default_factory=list)
    model_path: Optional[str] = None
    policy_sample: Optional[str] = None
    # Dungeon depth reached (SPD training only; None for the toy sim).
    avg_depth_trained: Optional[float] = None
    avg_depth_random: Optional[float] = None
    # Best single run seen (training or eval): deepest floor + the gear held there.
    best_depth: Optional[int] = None
    best_gear: Optional[str] = None

    def headline(self) -> str:
        return f"avg return {self.avg_return_trained:.1f} (random {self.avg_return_random:.1f})"


def train(
    env: GameEnv, agent: QLearningAgent, reward: RewardSpec, episodes: int,
    featurizer: Featurizer = _identity,
    epsilon_start: float = 1.0, epsilon_final: float = 0.05,
) -> list[float]:
    """Interact for `episodes`, learning from the user-defined reward.

    The agent learns over ``featurizer(obs)`` (a compact decision-state) while the
    reward is computed from the full observation. Returns a learning curve (mean
    episode return per ~5% block).

    Exploration decays from ``epsilon_start`` to ``epsilon_final``. A run that
    CONTINUES from an already-trained policy must lower ``epsilon_start`` — with
    the default 1.0 it would spend the first episodes acting at random and train
    the loaded network against that, degrading the very policy it resumed.
    """
    curve: list[float] = []
    window: list[float] = []
    block = max(1, episodes // 20)
    span = max(epsilon_start - epsilon_final, 0.0)
    for ep in range(episodes):
        epsilon = max(epsilon_final, epsilon_start - span * ep / max(1, episodes))
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(featurizer(obs), epsilon)
            nxt, done, info = env.step(action)
            r = reward.compute(obs, nxt, done, info)
            agent.learn(featurizer(obs), action, r, featurizer(nxt), done)
            total += r
            obs = nxt
        window.append(total)
        if len(window) >= block:
            curve.append(round(mean(window), 2))
            window = []
    if window:
        curve.append(round(mean(window), 2))
    return curve


def train_parallel(
    agent: QLearningAgent, envs: list[GameEnv], reward: RewardSpec, episodes: int,
    featurizer: Featurizer = _identity,
    epsilon_start: float = 1.0, epsilon_final: float = 0.05,
) -> list[float]:
    """Like `train`, but collect experience from several envs at once into one
    shared agent — so a single policy learns from many games in parallel.

    Each SPD env spends part of a step blocked on the game-to-learner pipe, and
    that wait releases the GIL, so stepping N envs in N threads overlaps that idle
    time. Only agent access (act/learn — the NumPy read/write of one network) is
    serialised with a lock; `env.step` (the I/O) runs lock-free. Returns the
    learning curve (per ~5% of total episodes), same shape as `train`.

    NOTE on speedup: this helps in proportion to how much of a step is spent
    WAITING on the game. With the compact focused-encoding observation the game
    computes a turn in well under a millisecond, so most of a step is GIL-held
    Python (JSON parse, featurize, the locked forward/backward) — measured ~1.3x
    on 4 envs, not 4x. For true N-core scaling the envs must run in separate
    PROCESSES (no shared GIL). Threads remain the right tool when a step is
    I/O-heavy (large observations / slow turns), where the overlap is large.
    """
    lock = threading.Lock()
    span = max(epsilon_start - epsilon_final, 0.0)
    block = max(1, episodes // 20)
    curve: list[float] = []
    window: list[float] = []
    state = {"done_eps": 0}

    def epsilon() -> float:
        return max(epsilon_final, epsilon_start - span * state["done_eps"] / max(1, episodes))

    def worker(env: GameEnv) -> None:
        while True:
            with lock:
                if state["done_eps"] >= episodes:
                    return
                eps = epsilon()
            obs = env.reset()
            fobs = featurizer(obs)               # pure work stays OUT of the lock
            done, total = False, 0.0
            while not done:
                with lock:                       # network read only
                    action = agent.act(fobs, eps)
                nxt, done, info = env.step(action)   # pipe I/O — lock-free, overlaps
                r = reward.compute(obs, nxt, done, info)
                fnxt = featurizer(nxt)
                with lock:                       # buffer append + periodic network write
                    agent.learn(fobs, action, r, fnxt, done)
                total += r
                obs, fobs = nxt, fnxt
            with lock:
                state["done_eps"] += 1
                window.append(total)
                if len(window) >= block:
                    curve.append(round(mean(window), 2))
                    window.clear()

    threads = [threading.Thread(target=worker, args=(e,), daemon=True) for e in envs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if window:
        curve.append(round(mean(window), 2))
    return curve


def evaluate(
    env: GameEnv, policy: Policy, reward: RewardSpec, episodes: int,
    featurizer: Featurizer = _identity,
) -> tuple[float, float]:
    """Run greedy/eval episodes; return (mean return, mean survived steps)."""
    returns, survivals = [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = policy(featurizer(obs))
            nxt, done, info = env.step(action)
            total += reward.compute(obs, nxt, done, info)
            obs = nxt
            steps += 1
        returns.append(total)
        survivals.append(steps)
    return mean(returns), mean(survivals)
