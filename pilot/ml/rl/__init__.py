"""Reinforcement-learning foundation for screenshot-driven games.

The agent learns over a **structured observation** — the "consistent important
data" extracted from each frame (player health, enemies nearby, ...), not raw
pixels — and the reward comes from the user's **good/bad events** (`RewardSpec`).

What's here is validated on a simulated game (`SimEnv`) so it provably learns and
is testable offline; the real Steam game plugs into the same `GameEnv` interface
via `capture.py` (screenshot -> feature extractor -> key/mouse actions).
"""

from __future__ import annotations

from .agent import QLearningAgent
from .env import GameEnv, Observation, SimEnv
from .reward import RewardRule, RewardSpec

__all__ = ["GameEnv", "Observation", "SimEnv", "RewardRule", "RewardSpec", "QLearningAgent"]
