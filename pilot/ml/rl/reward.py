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
    # Multiply the weight by the CURRENT value of this field, so a reward can
    # compound instead of paying a flat rate. Needed where later progress is
    # genuinely worth more than earlier progress (in a dungeon crawler, reaching
    # floor 10 is worth far more than reaching floor 2 — a flat rate tells the
    # agent depth doesn't compound, and it will happily farm the first floor).
    # One field name, or several — the weight is multiplied by each field's
    # current value, so a reward can depend on more than one thing at once
    # (e.g. descending pays by how DEEP you now are AND how thoroughly you
    # worked the floor you left, making a dive past the loot worth ~nothing).
    scale_by: Optional[str | list[str]] = None
    # Upper bound on the scale multiplier. A ramp that grows without limit can
    # dominate every other term and, worse, blow up the TD targets a value
    # network is trying to fit — so a ramp should climb and then SATURATE.
    scale_cap: Optional[float] = None

    def value(self, prev: Observation, cur: Observation) -> float:
        if self.max_depth is not None and cur.get("depth", 0.0) > self.max_depth:
            return 0.0
        d = cur.get(self.field, 0.0) - prev.get(self.field, 0.0)
        weight = self.weight
        if self.scale_by is not None:
            fields = [self.scale_by] if isinstance(self.scale_by, str) else self.scale_by
            scale = 1.0
            for f in fields:
                scale *= cur.get(f, 1.0)
            if self.scale_cap is not None:
                scale = min(scale, self.scale_cap)
            weight *= scale
        if self.direction == "up" and d > 0:
            return weight * (d if self.per_unit else 1.0)
        if self.direction == "down" and d < 0:
            return weight * (abs(d) if self.per_unit else 1.0)
        return 0.0


class RewardSpec(BaseModel):
    """A bundle of good/bad rules plus a survival bonus and a death penalty."""

    rules: list[RewardRule] = Field(default_factory=list)
    step_reward: float = 0.0          # small bonus per surviving step
    death_field: str = "player_health"
    death_threshold: float = 0.0
    death_reward: float = 0.0         # applied once when done & field <= threshold
    # Per-turn costs charged when a flag field is truthy in the resulting
    # observation, e.g. the action accomplished nothing, or the turn was spent
    # idling with nothing to gain. Charging only on a flag (rather than every
    # step) lets genuinely useful "quiet" turns — resting off damage — stay free.
    flag_penalties: dict[str, float] = Field(default_factory=dict)

    def compute(self, prev: Observation, cur: Observation, done: bool, info: dict) -> float:
        r = self.step_reward
        for rule in self.rules:
            r += rule.value(prev, cur)
        for field, penalty in self.flag_penalties.items():
            if cur.get(field, 0.0) >= 1.0:
                r += penalty
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
