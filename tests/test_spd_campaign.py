"""Campaign/curriculum tests: stage program, challenge masks, hero+challenge
protocol, and the win rewards — no JVM needed (fake process)."""

from __future__ import annotations

import pytest

from pilot.ml.rl.spd import spd_reward_spec
from pilot.ml.rl.spd_campaign import Stage, challenge_mask, default_stages
from pilot.ml.rl.spd_real import SPDRealEnv

from test_spd_real import FakeProc, _obs_line


def test_challenge_mask_progression():
    assert challenge_mask(0) == 0
    assert challenge_mask(1) == 1          # NO_FOOD
    assert challenge_mask(2) == 3          # + NO_ARMOR
    assert challenge_mask(9) == 511        # all nine
    with pytest.raises(ValueError):
        challenge_mask(10)


def test_default_stages_cover_the_request():
    stages = default_stages(episodes=1000)
    assert stages[0] == Stage("baseline-warrior", "warrior", 0, 2000)   # long baseline
    heroes = [s.hero for s in stages[1:6]]
    assert heroes == ["mage", "rogue", "huntress", "duelist", "cleric"]
    ladder = [s.challenges for s in stages[6:]]
    assert ladder == list(range(1, 10))                                  # 1..9 challenges
    assert all(s.hero == "warrior" for s in stages[6:])


def test_reset_sends_hero_and_challenge_mask():
    proc = FakeProc([_obs_line()])
    env = SPDRealEnv(seed=50, proc=proc, hero="mage", challenges=challenge_mask(3))
    env.reset()
    assert proc.commands == ["reset 50 mage 7"]


def test_unknown_hero_rejected():
    with pytest.raises(ValueError, match="unknown hero"):
        SPDRealEnv(proc=FakeProc([]), hero="paladin")


def test_amulet_and_win_are_rewarded():
    reward = spd_reward_spec()
    base = {"hp_current": 20.0, "has_amulet": 0.0, "won": 0.0}
    amulet = dict(base, has_amulet=1.0)
    won = dict(amulet, won=1.0)
    r_amulet = reward.compute(base, amulet, False, {})
    r_win = reward.compute(amulet, won, True, {})
    assert r_amulet >= 200
    assert r_win >= 500


def test_step_reports_win_in_info():
    proc = FakeProc([_obs_line(), _obs_line(has_amulet=1, won=1, done=True)])
    env = SPDRealEnv(proc=proc)
    env.reset()
    obs, done, info = env.step("descend")
    assert done and info["won"] and obs["has_amulet"] == 1
