"""Game environment interface + a simulated game to validate learning.

An `Observation` is the structured data extracted from a frame (e.g.
``{"player_health": 80, "enemies_nearby": 2, "enemy_adjacent": 1}``). A
`GameEnv` exposes ``reset`` / ``step`` / ``action_space``; the reward is NOT
returned by the env — it is computed from observation changes by a
``RewardSpec`` (so the user owns the good/bad definition).

`SimEnv` is a tiny survival game: enemies approach and deal damage; the agent
can attack (clear an enemy), dodge (likely avoid damage), heal (limited), or
wait. It's deterministic given a seed, so training and tests are reproducible.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

Observation = dict[str, float]


class GameEnv(ABC):
    """Minimal environment contract shared by the sim and the real game."""

    action_space: list[str]

    @abstractmethod
    def reset(self) -> Observation:
        ...

    @abstractmethod
    def step(self, action: str) -> tuple[Observation, bool, dict[str, Any]]:
        """Apply an action; return (observation, done, info). No reward here."""
        ...

    def observation_fields(self) -> list[str]:
        return []

    def discretization(self) -> dict[str, list[float]]:
        """Bin edges for continuous fields (for tabular agents). Default: none."""
        return {}


class SimEnv(GameEnv):
    """A small, learnable survival game for validating the RL stack offline."""

    action_space = ["attack", "dodge", "heal", "wait"]

    def __init__(self, seed: int = 0, max_steps: int = 60, max_enemies: int = 3):
        self._rng = random.Random(seed)
        self.max_steps = max_steps
        self.max_enemies = max_enemies
        self.health = 100.0
        self.enemies = 0
        self.heals_left = 3
        self.steps = 0

    def observation_fields(self) -> list[str]:
        return ["player_health", "enemies_nearby", "enemy_adjacent"]

    def discretization(self) -> dict[str, list[float]]:
        return {"player_health": [20.0, 40.0, 60.0, 80.0]}

    def _obs(self) -> Observation:
        return {
            "player_health": float(self.health),
            "enemies_nearby": float(self.enemies),
            "enemy_adjacent": 1.0 if self.enemies > 0 else 0.0,
        }

    def reset(self) -> Observation:
        self.health = 100.0
        self.enemies = 0
        self.heals_left = 3
        self.steps = 0
        return self._obs()

    def step(self, action: str) -> tuple[Observation, bool, dict[str, Any]]:
        self.steps += 1
        defeated = 0

        if action == "attack" and self.enemies > 0:
            self.enemies -= 1
            defeated = 1
        elif action == "heal" and self.heals_left > 0:
            self.health = min(100.0, self.health + 25.0)
            self.heals_left -= 1

        # An enemy may approach.
        if self.enemies < self.max_enemies and self._rng.random() < 0.35:
            self.enemies += 1

        # Take damage if any enemy is present, unless a dodge succeeds.
        if self.enemies > 0:
            dodged = action == "dodge" and self._rng.random() < 0.7
            if not dodged:
                self.health -= 12.0 * self.enemies
        self.health = max(0.0, self.health)

        done = self.health <= 0 or self.steps >= self.max_steps
        info = {"defeated": defeated, "survived": self.steps >= self.max_steps and self.health > 0}
        return self._obs(), done, info
