"""Campaign/curriculum tests: stage program, challenge masks, hero+challenge
protocol, and the win rewards — no JVM needed (fake process)."""

from __future__ import annotations

import pytest

from pilot.ml.rl.spd import SPD_ACTIONS, spd_reward_spec
from pilot.ml.rl.spd_campaign import Stage, challenge_mask, default_stages
from pilot.ml.rl.spd_real import SPDRealEnv
from pilot.ml.rl.spd_sim import spd_featurizer

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


def test_capability_actions_and_features():
    # capabilities exist as actions; judgment stays with the agent
    for action in ("eat_food", "zap_wand", "equip_gear"):
        assert action in SPD_ACTIONS and action in SPDRealEnv.action_space
    # gear progression is FOUR separate decisions (no bundling macro), so the
    # agent learns spend-now vs save-up rather than the engine forcing an order
    for action in ("equip_gear", "equip_misc", "drink_strength", "read_upgrade"):
        assert action in SPD_ACTIONS and action in SPDRealEnv.action_space
    # talent allocation: the agent picks which perk slot to invest (its build)
    for action in ("talent_0", "talent_1", "talent_2", "talent_3"):
        assert action in SPD_ACTIONS and action in SPDRealEnv.action_space
    # capability state is featurized (learnable), defaulting to 0 for the sim
    feat = spd_featurizer({"hp_bin": 4, "enemies_visible": 1, "enemy_dir": 3,
                           "enemy_adjacent": 0, "stairs_dir": 0, "has_heal": 0,
                           "starving": 0, "has_food": 1, "wand_charges": 5,
                           "gear_available": 1, "challenge_count": 4})
    assert feat["has_food"] == 1 and feat["gear_available"] == 1
    assert feat["challenge_count"] == 4
    assert feat["wand_charges"] == 2          # capped: big pools bin together
    sim_feat = spd_featurizer({"hp_bin": 4, "enemies_visible": 0, "enemy_dir": 0,
                               "enemy_adjacent": 0, "stairs_dir": 0, "has_heal": 0,
                               "starving": 0})
    assert sim_feat["has_food"] == 0          # sim lacks the field -> default 0


def test_gated_campaign_advances_only_on_wins():
    # fake env: hero A wins base game and challenge 1, then never again;
    # verifies the gate, the retry, and the no-jumps halt.
    from pilot.ml.rl import spd_campaign as sc

    class FakeEnv:
        action_space = list(SPD_ACTIONS)
        HEROES = ("warrior",)
        calls = []

        def __init__(self, seed=0, max_steps=0, hero="warrior", challenges=0, proc=None):
            self.challenges = challenges
            self.steps = 0
            FakeEnv.calls.append(challenges)

        def reset(self):
            self.steps = 0
            return {"hp_current": 20.0}

        def step(self, action):
            self.steps += 1
            done = self.steps >= 3
            # win instantly at challenge masks 0 and 1; never above
            won = done and self.challenges <= 1
            return ({"hp_current": 20.0, "won": 1.0 if won else 0.0},
                    done, {"depth": 1, "won": won})

        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *e): pass

    orig = sc.SPDRealEnv
    sc.SPDRealEnv = FakeEnv
    try:
        tracks, ws = sc.run_gated_campaign(
            episodes=2, heroes=("warrior",), max_attempts=2, eval_episodes=2)
    finally:
        sc.SPDRealEnv = orig

    assert len(tracks) == 1
    t = tracks[0]
    assert t.reached == 1                      # beat base + challenge 1
    assert t.attempts["warrior-chal2"] == 2    # retried the failed gate
    assert max(s.challenges for s in t.stages) == 2   # never jumped past the wall
