"""RL game-agent tests — fully offline, deterministic (no game, no MCP).

Validates the simulated env, the user-defined reward, that the agent actually
learns (beats random), the end-to-end runner, and the real-game seam interfaces.
"""

from __future__ import annotations

import random
from pathlib import Path

from pilot.ml.rl.agent import QLearningAgent
from pilot.ml.rl.capture import ActionDriver, Capturer, FeatureExtractor, ScreenGameEnv
from pilot.ml.rl.env import SimEnv
from pilot.ml.rl.reward import RewardRule, RewardSpec
from pilot.ml.rl.runner import run_rl_goal


# ------------------------------------------------------------------- env
def test_simenv_dynamics():
    env = SimEnv(seed=0)
    obs = env.reset()
    assert obs["player_health"] == 100.0 and obs["enemies_nearby"] == 0.0
    assert set(env.observation_fields()) == {"player_health", "enemies_nearby", "enemy_adjacent"}
    # A purely-passive player eventually dies or the episode ends.
    done = False
    steps = 0
    while not done and steps < 200:
        obs, done, info = env.step("wait")
        steps += 1
    assert done


# ------------------------------------------------------------------- reward
def test_reward_spec_rules():
    spec = RewardSpec.survival_default()
    # defeating an enemy (enemies_nearby down) is good
    r = spec.compute({"enemies_nearby": 2, "player_health": 100},
                     {"enemies_nearby": 1, "player_health": 100}, False, {})
    assert r > 0
    # taking damage (player_health down) is bad
    r = spec.compute({"enemies_nearby": 0, "player_health": 100},
                     {"enemies_nearby": 0, "player_health": 80}, False, {})
    assert r < spec.step_reward
    # dying applies the death penalty
    r = spec.compute({"player_health": 10, "enemies_nearby": 1},
                     {"player_health": 0, "enemies_nearby": 1}, True, {})
    assert r <= spec.death_reward

    rule = RewardRule(field="score", direction="up", weight=2.0, per_unit=True)
    assert rule.value({"score": 10}, {"score": 15}) == 10.0  # +2 * delta(5)


# ------------------------------------------------------------------- agent learns
def test_agent_learns_to_beat_random(tmp_path):
    result, ws = run_rl_goal(episodes=3000, base_dir=tmp_path / "rl", seed=0)
    # The trained policy should clearly outperform random and survive longer.
    assert result.avg_return_trained > result.avg_return_random
    assert result.improvement > 2.0
    assert result.avg_survival_trained > result.avg_survival_random
    assert result.states_learned > 0
    # Artifacts + report written to the workspace.
    assert Path(result.model_path).exists()
    assert (ws.path / "report.md").exists()
    assert (ws.path / "metrics.json").exists()
    assert (ws.path / "reward_spec.json").exists()


# ------------------------------------------------------------------- real-game seam
def test_screen_game_env_with_fakes():
    class FakeCap(Capturer):
        def grab(self):
            return "frame"

    class FakeExt(FeatureExtractor):
        def __init__(self):
            self.hp = 100

        def extract(self, frame):
            self.hp -= 25
            return {"player_health": float(self.hp), "enemies_nearby": 1.0}

    class FakeDrv(ActionDriver):
        def __init__(self):
            self.calls = []

        def do(self, action):
            self.calls.append(action)

    drv = FakeDrv()
    env = ScreenGameEnv(FakeCap(), FakeExt(), drv, ["attack", "wait"],
                        done_fn=lambda o: o["player_health"] <= 0)
    obs = env.reset()
    assert "player_health" in obs
    done = False
    while not done:
        obs, done, info = env.step("attack")
    assert drv.calls and all(a == "attack" for a in drv.calls)
    assert obs["player_health"] <= 0
