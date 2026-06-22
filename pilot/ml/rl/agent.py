"""A tabular Q-learning agent over the structured observation.

Continuous fields (e.g. player_health) are binned via the env's discretization
hints; small integer fields pass through. Keeps the state space tiny so the
agent learns fast with no deep-learning dependency. This is the swappable
"learner" — a DQN could replace it later for larger state spaces.
"""

from __future__ import annotations

import bisect
import random
from typing import Optional

from .env import Observation


class QLearningAgent:
    def __init__(
        self,
        actions: list[str],
        bins: Optional[dict[str, list[float]]] = None,
        alpha: float = 0.2,
        gamma: float = 0.95,
        seed: int = 0,
    ):
        self.actions = list(actions)
        self.bins = bins or {}
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.Q: dict[tuple, float] = {}

    def state_key(self, obs: Observation) -> tuple:
        """Deterministic, discretized key from the observation."""
        parts = []
        for field in sorted(obs):
            v = obs[field]
            if field in self.bins:
                v = bisect.bisect_right(self.bins[field], v)
            else:
                v = int(round(v))
            parts.append(v)
        return tuple(parts)

    def _greedy(self, obs: Observation) -> str:
        s = self.state_key(obs)
        best, best_v = [], -1e18
        for a in self.actions:
            v = self.Q.get((s, a), 0.0)
            if v > best_v:
                best_v, best = v, [a]
            elif v == best_v:
                best.append(a)
        return self.rng.choice(best)

    def act(self, obs: Observation, epsilon: float = 0.0) -> str:
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.choice(self.actions)
        return self._greedy(obs)

    def policy(self, obs: Observation) -> str:
        """Greedy action (for evaluation / deployment)."""
        return self._greedy(obs)

    def learn(self, obs: Observation, action: str, reward: float, nxt: Observation, done: bool) -> None:
        s, s2 = self.state_key(obs), self.state_key(nxt)
        future = 0.0 if done else max(self.Q.get((s2, a), 0.0) for a in self.actions)
        key = (s, action)
        self.Q[key] = self.Q.get(key, 0.0) + self.alpha * (reward + self.gamma * future - self.Q.get(key, 0.0))

    @property
    def states_learned(self) -> int:
        return len({s for s, _ in self.Q})
