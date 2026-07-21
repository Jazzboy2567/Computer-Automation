"""Reward = the user's good/bad events, expressed as rules over observation changes.

This is where "I can list what good and bad things can happen" lives. A rule
fires when a field goes up or down between frames; a terminal `death` condition
adds a one-off penalty. The env never hardcodes reward — this does — so the same
agent works on any game once you describe its good/bad events.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .env import Observation


class RewardRule(BaseModel):
    """Fire `weight` when `field` moves in `direction` between two frames."""

    field: str
    direction: Literal["up", "down"]
    weight: float
    per_unit: bool = False  # scale weight by the magnitude of the change
    # Optional depth gate: the rule is inert once `depth` exceeds this. Used to
    # confine an early-game incentive (e.g. "try new gear on") to shallow floors
    # so it doesn't push behaviour the player wouldn't want deep (swapping a
    # settled build). Inert where the observation has no `depth`.
    max_depth: Optional[float] = None

    def value(self, prev: Observation, cur: Observation) -> float:
        if self.max_depth is not None and cur.get("depth", 0.0) > self.max_depth:
            return 0.0
        d = cur.get(self.field, 0.0) - prev.get(self.field, 0.0)
        if self.direction == "up" and d > 0:
            return self.weight * (d if self.per_unit else 1.0)
        if self.direction == "down" and d < 0:
            return self.weight * (abs(d) if self.per_unit else 1.0)
        return 0.0


class RewardSpec(BaseModel):
    """A bundle of good/bad rules plus a survival bonus and a death penalty."""

    rules: list[RewardRule] = Field(default_factory=list)
    step_reward: float = 0.0          # small bonus per surviving step
    death_field: str = "player_health"
    death_threshold: float = 0.0
    death_reward: float = 0.0         # applied once when done & field <= threshold
    # A per-step cost when the chosen action accomplished nothing (`waste_field`
    # is truthy in the resulting observation). Keeps an impossible action from
    # being a "free" no-op a value-overestimating policy can collapse onto.
    waste_field: str = ""
    waste_penalty: float = 0.0

    def compute(self, prev: Observation, cur: Observation, done: bool, info: dict) -> float:
        r = self.step_reward
        for rule in self.rules:
            r += rule.value(prev, cur)
        if self.waste_field and cur.get(self.waste_field, 0.0) >= 1.0:
            r += self.waste_penalty
        if done and cur.get(self.death_field, self.death_threshold + 1.0) <= self.death_threshold:
            r += self.death_reward
        return r

    @staticmethod
    def survival_default() -> "RewardSpec":
        """A sensible survival reward (used by the simulated demo)."""
        return RewardSpec(
            rules=[
                RewardRule(field="enemies_nearby", direction="down", weight=3.0),       # defeated an enemy = good
                RewardRule(field="player_health", direction="down", weight=-0.1, per_unit=True),  # taking damage = bad
            ],
            step_reward=0.5,        # surviving is good
            death_reward=-10.0,     # dying is very bad
        )
