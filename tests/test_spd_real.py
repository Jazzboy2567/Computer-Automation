"""SPDRealEnv protocol tests with a fake Java process (no JVM needed), plus an
integration test that runs only when the compiled bridge is present."""

from __future__ import annotations

import json

import pytest

from pilot.ml.rl.spd_real import (
    UNKNOWN_STAIRS_DIST, SPDRealEnv, bridge_available,
)
from pilot.ml.rl.spd_sim import spd_featurizer


def _obs_line(**over) -> str:
    base = {
        "hp_current": 20, "hp_max": 20, "hp_frac": 1.0, "level": 1, "xp_frac": 0,
        "depth": 1, "gold": 0, "enemies_visible": 0, "inventory_count": 4,
        "starving": 0, "has_ankh": 0, "hp_bin": 4, "enemy_dir": 0,
        "enemy_adjacent": 0, "stairs_dir": 0, "stairs_dist": -1, "has_heal": 0,
        "cells_explored": 60, "has_amulet": 0, "won": 0,
        "turns": 0, "pos": 159, "done": False,
    }
    base.update(over)
    return json.dumps(base) + "\n"


class FakeProc:
    """Mimics the EnvServer: records commands, replies with scripted lines."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.commands = []
        self.proc_self = self

        class _In:
            def __init__(self, outer): self.outer = outer
            def write(self, s): self.outer.commands.append(s.strip())
            def flush(self): pass

        class _Out:
            def __init__(self, outer): self.outer = outer
            def readline(self): return self.outer.replies.pop(0) if self.outer.replies else ""

        self.stdin = _In(self)
        self.stdout = _Out(self)

    def poll(self): return None
    def wait(self, timeout=None): return 0
    def kill(self): pass


def test_reset_maps_observation_and_seeds():
    proc = FakeProc([_obs_line(), _obs_line()])
    env = SPDRealEnv(seed=100, proc=proc)
    obs = env.reset()
    assert proc.commands == ["reset 100 warrior 0"]
    assert obs["hp_current"] == 20 and obs["hp_bin"] == 4
    assert "done" not in obs and "turns" not in obs and "pos" not in obs
    env.reset()
    assert proc.commands[-1] == "reset 101 warrior 0"    # fresh dungeon per episode


def test_unknown_stairs_maps_to_assumed_far():
    proc = FakeProc([_obs_line(stairs_dist=-1), _obs_line(stairs_dist=7)])
    env = SPDRealEnv(proc=proc)
    assert env.reset()["stairs_dist"] == UNKNOWN_STAIRS_DIST
    obs, _, _ = env.step("move_n")
    assert obs["stairs_dist"] == 7               # discovered: true distance


def test_step_done_and_max_steps():
    proc = FakeProc([_obs_line(), _obs_line(), _obs_line(done=True, hp_current=0)])
    env = SPDRealEnv(proc=proc, max_steps=50)
    env.reset()
    obs, done, info = env.step("move_e")
    assert not done and info["depth"] == 1
    obs, done, _ = env.step("attack_nearest")
    assert done and obs["hp_current"] == 0       # death ends the episode

    proc2 = FakeProc([_obs_line()] + [_obs_line()] * 3)
    env2 = SPDRealEnv(proc=proc2, max_steps=2)
    env2.reset()
    _, done, _ = env2.step("wait")
    assert not done
    _, done, _ = env2.step("wait")
    assert done                                   # step cap ends the episode


def test_featurizer_accepts_real_observation():
    proc = FakeProc([_obs_line()])
    env = SPDRealEnv(proc=proc)
    feat = spd_featurizer(env.reset())
    assert set(feat) == {"hp_bin", "enemies_visible", "enemy_dir",
                         "enemy_adjacent", "stairs_dir", "has_heal", "starving",
                         "has_food", "wand_charges", "gear_available", "challenge_count"}


def test_explore_in_action_space_and_shaping_rewards_new_cells():
    from pilot.ml.rl.spd import spd_training_reward

    assert "explore" in SPDRealEnv.action_space
    reward = spd_training_reward()
    prev = {"cells_explored": 60.0, "hp_current": 20.0}
    cur = {"cells_explored": 80.0, "hp_current": 20.0}
    gained = reward.compute(prev, cur, False, {})
    flat = reward.compute(prev, dict(prev), False, {})
    assert gained > flat                          # seeing new floor pays
    # the sim never emits the field -> the rule must be inert there
    assert reward.compute({"hp_current": 20.0}, {"hp_current": 20.0}, False, {}) == flat


def test_close_sends_quit():
    proc = FakeProc([_obs_line()])
    env = SPDRealEnv(proc=proc)
    env.reset()
    env.close()
    assert proc.commands[-1] == "quit"


def test_server_error_raises():
    proc = FakeProc(['{"error":"NullPointerException"}\n'])
    env = SPDRealEnv(proc=proc)
    with pytest.raises(RuntimeError, match="NullPointerException"):
        env.reset()


@pytest.mark.skipif(not bridge_available(), reason="SPD clone/bridge not built here")
def test_integration_real_game_steps():
    """Ten real actions against the actual game (only when the bridge exists)."""
    with SPDRealEnv(seed=777, max_steps=20) as env:
        obs = env.reset()
        assert obs["hp_current"] > 0 and obs["depth"] == 1
        for action in ["move_e", "explore", "wait", "search", "attack_nearest",
                       "pickup", "descend", "use_item", "explore", "move_n"]:
            obs, done, info = env.step(action)
            assert "hp_current" in obs
            if done:
                break
        assert info["turns"] > 0                  # game time actually advanced
