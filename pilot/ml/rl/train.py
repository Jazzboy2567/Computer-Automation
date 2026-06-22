"""Training loop + evaluation for the RL agent.

Reward is computed from observation changes by the `RewardSpec`, not the env, so
the same loop drives the simulated game and (later) the real one.
"""

from __future__ import annotations

from statistics import mean
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .agent import QLearningAgent
from .env import GameEnv, Observation
from .reward import RewardSpec

Policy = Callable[[Observation], str]


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

    def headline(self) -> str:
        return f"avg return {self.avg_return_trained:.1f} (random {self.avg_return_random:.1f})"


def train(env: GameEnv, agent: QLearningAgent, reward: RewardSpec, episodes: int) -> list[float]:
    """Interact for `episodes`, learning from the user-defined reward.

    Returns a learning curve (mean episode return per ~5% block).
    """
    curve: list[float] = []
    window: list[float] = []
    block = max(1, episodes // 20)
    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - ep / max(1, episodes))  # explore -> exploit
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, epsilon)
            nxt, done, info = env.step(action)
            r = reward.compute(obs, nxt, done, info)
            agent.learn(obs, action, r, nxt, done)
            total += r
            obs = nxt
        window.append(total)
        if len(window) >= block:
            curve.append(round(mean(window), 2))
            window = []
    if window:
        curve.append(round(mean(window), 2))
    return curve


def evaluate(env: GameEnv, policy: Policy, reward: RewardSpec, episodes: int) -> tuple[float, float]:
    """Run greedy/eval episodes; return (mean return, mean survived steps)."""
    returns, survivals = [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = policy(obs)
            nxt, done, info = env.step(action)
            total += reward.compute(obs, nxt, done, info)
            obs = nxt
            steps += 1
        returns.append(total)
        survivals.append(steps)
    return mean(returns), mean(survivals)
