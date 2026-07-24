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
    assert proc.commands == ["reset 100 warrior 0 1"]    # trailing arg = start floor
    assert obs["hp_current"] == 20 and obs["hp_bin"] == 4
    assert "done" not in obs and "turns" not in obs and "pos" not in obs
    env.reset()
    assert proc.commands[-1] == "reset 101 warrior 0 1"  # fresh dungeon per episode


def test_curriculum_starts_deeper_but_does_not_inflate_best_depth():
    # a curriculum episode BEGINS deep, so it must not count as "reached floor 4"
    proc = FakeProc([_obs_line(depth=4), _obs_line(depth=4)])
    env = SPDRealEnv(seed=1, proc=proc, curriculum=lambda ep: 4)
    env.reset()
    assert proc.commands[-1] == "reset 1 warrior 0 4"
    env.step("wait")
    assert env.best_depth == 0                            # not an achievement

    proc2 = FakeProc([_obs_line(depth=1), _obs_line(depth=3)])
    env2 = SPDRealEnv(seed=1, proc=proc2)                 # no curriculum -> real run
    env2.reset()
    env2.step("descend")
    assert env2.best_depth == 3                           # genuinely reached


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
                         "has_food", "wand_charges", "gear_available", "challenge_count",
                         "enemy_unaware", "has_missile", "loot_here", "has_bow",
                         "has_unknown_potion", "has_unknown_scroll",
                         "str_potions", "upgrade_scrolls", "cursed_equipped",
                         "misc_available", "frontier_left", "loot_visible"}


def test_explore_is_a_capability_not_a_paid_objective():
    """`explore` stays in the action space (it's how the agent covers ground),
    but covering ground is not itself rewarded — only what it FINDS is."""
    from pilot.ml.rl.spd import spd_training_reward

    assert "explore" in SPDRealEnv.action_space
    reward = spd_training_reward()
    prev = {"explored_frac": 0.2, "hp_current": 20.0, "inventory_count": 4.0}
    flat = reward.compute(prev, dict(prev), False, {})
    explored = reward.compute(prev, {**prev, "explored_frac": 0.9}, False, {})
    assert explored == flat                       # ground covered, nothing found
    found = reward.compute(prev, {**prev, "explored_frac": 0.9,
                                  "inventory_count": 5.0}, False, {})
    assert found > flat                           # exploring PAYS via what it finds


def test_identifying_an_item_type_is_rewarded():
    from pilot.ml.rl.spd import spd_training_reward

    reward = spd_training_reward()
    prev = {"known_item_types": 2.0, "hp_current": 20.0}
    flat = reward.compute(prev, dict(prev), False, {})
    identified = reward.compute(prev, {"known_item_types": 3.0, "hp_current": 20.0}, False, {})
    assert identified > flat                       # learning what an item does pays
    # surviving the gamble (even at some HP cost) still nets positive — that's the
    # point, the agent should be WILLING to drink the mystery potion
    hurt = reward.compute(prev, {"known_item_types": 3.0, "hp_current": 12.0}, False, {})
    assert hurt > flat
    # but a potion that IDs itself by KILLING you is dominated by the death
    # penalty, so recklessness at low HP is still punished — nuance the agent must learn
    lethal = reward.compute(prev, {"known_item_types": 3.0, "hp_current": 0.0}, True, {})
    assert lethal < flat
    # inert where the field is absent (the sim never emits it)
    assert reward.compute({"hp_current": 20.0}, {"hp_current": 20.0}, False, {}) == flat


def test_gear_improvement_rewarded_only_early():
    from pilot.ml.rl.spd import spd_training_reward

    reward = spd_training_reward()
    # equipping a better weapon early (floor 2): gear_score rises -> pays
    early = reward.compute({"gear_score": 1.0, "hp_current": 20.0, "depth": 2.0},
                           {"gear_score": 3.0, "hp_current": 20.0, "depth": 2.0}, False, {})
    flat = reward.compute({"gear_score": 1.0, "hp_current": 20.0, "depth": 2.0},
                          {"gear_score": 1.0, "hp_current": 20.0, "depth": 2.0}, False, {})
    assert early > flat
    # the SAME improvement deep (floor 12) is inert — no reward to churn a build
    late = reward.compute({"gear_score": 1.0, "hp_current": 20.0, "depth": 12.0},
                          {"gear_score": 3.0, "hp_current": 20.0, "depth": 12.0}, False, {})
    late_flat = reward.compute({"gear_score": 1.0, "hp_current": 20.0, "depth": 12.0},
                               {"gear_score": 1.0, "hp_current": 20.0, "depth": 12.0}, False, {})
    assert late == late_flat


def test_descending_outpays_farming_a_floor_and_compounds():
    """The failure that produced 21 runs stuck on floor ~1.9 in starting gear:
    fully exploring a floor paid more than the stairs, so farming floor 1 was
    optimal. Descending must beat it, and must compound with depth."""
    from pilot.ml.rl.spd import spd_training_reward

    reward = spd_training_reward()
    base = {"hp_current": 20.0, "depth": 1.0, "explored_frac": 0.0,
            "inventory_count": 4.0}
    nothing = reward.compute(base, dict(base), False, {})

    # covering ground is NOT the objective: walking a whole floor and finding
    # nothing pays exactly the same as standing still
    walked = reward.compute(base, {**base, "explored_frac": 1.0}, False, {})
    assert walked == nothing, "exploring must not be paid for in itself"

    # picking items up IS the objective
    looted = reward.compute(base, {**base, "inventory_count": 6.0}, False, {})
    assert looted > nothing

    # and depth compounds: deeper descents are worth strictly more (comparing
    # like for like — both after working the floor over)
    worked = {"descent_explored": 1.0}
    descend = reward.compute(base, {**base, "depth": 2.0, **worked}, False, {})
    deep_prev = {**base, "depth": 8.0}
    deep = reward.compute(deep_prev, {**deep_prev, "depth": 9.0, **worked}, False, {})
    assert deep > descend * 2, "reaching floor 9 must be worth far more than floor 2"


def test_diving_past_a_floors_loot_pays_almost_nothing():
    """Depth only pays for a floor you actually worked. Otherwise the fastest way
    to bank the compounding descent reward is to stair-dive straight past every
    floor's loot — the opposite of the early-game play we want."""
    from pilot.ml.rl.spd import spd_reward_spec

    reward = spd_reward_spec()
    base = {"hp_current": 20.0, "depth": 1.0, "inventory_count": 4.0}

    dive = reward.compute(base, {**base, "depth": 2.0, "descent_explored": 0.05}, False, {})
    cleared = reward.compute(base, {**base, "depth": 2.0, "descent_explored": 1.0}, False, {})
    assert cleared > 10 * dive, "diving past the loot must be worth far less"
    assert dive < 1.0, "an untouched floor's descent is worth ~nothing"


def test_getting_stuck_costs_progressively_more():
    """Progress (a fight, a heal, new ground, loot, a floor change) keeps the
    stall streak at 0 and is free. Once a floor is picked clean and quiet the
    streak climbs, and the cost must RAMP — no magic cutoff — so lingering can
    never beat descending (the agent sat out 415 of 600 turns at depth 1.17)."""
    from pilot.ml.rl.spd import spd_reward_spec

    reward = spd_reward_spec()
    base = {"hp_current": 20.0, "depth": 1.0, "stall_streak": 0.0}

    # progress keeps it free
    assert reward.compute(base, dict(base), False, {}) == 0.0

    # each further stalled turn costs strictly more than the last
    step5 = reward.compute({**base, "stall_streak": 4.0},
                           {**base, "stall_streak": 5.0}, False, {})
    step50 = reward.compute({**base, "stall_streak": 49.0},
                            {**base, "stall_streak": 50.0}, False, {})
    assert step5 < 0 and step50 < step5, "the stall cost must ramp, not stay flat"

    # and a long stall must end up worse than the reward for descending a floor
    total = sum(
        reward.compute({**base, "stall_streak": float(n)},
                       {**base, "stall_streak": float(n + 1)}, False, {})
        for n in range(60)
    )
    descend = reward.compute(base, {**base, "depth": 2.0,
                                    "descent_explored": 1.0}, False, {})
    assert total < -descend, "stalling a floor out must cost more than a descent pays"

    # but the ramp must SATURATE, not grow without bound: an unbounded ramp let a
    # stuck policy bleed ~-125 a run, drowning every other signal and blowing up
    # the value targets the network fits
    step200 = reward.compute({**base, "stall_streak": 199.0},
                             {**base, "stall_streak": 200.0}, False, {})
    assert step200 == step50, "the stall cost must plateau once it is clearly bad"


def test_trying_a_descent_beats_idling_the_clock_out():
    """Even a descent that ends in death must beat stalling out the clock, or an
    agent that cannot yet fight will correctly refuse to ever take the stairs —
    and so can never practise the depths where it would learn to survive."""
    from pilot.ml.rl.spd import spd_reward_spec

    reward = spd_reward_spec()
    steps, start = 600, {"hp_current": 20.0, "depth": 1.0, "inventory_count": 4.0}

    # (a) stall out the whole episode on floor 1 (nothing found, nothing fought)
    idle = sum(reward.compute({**start, "stall_streak": float(n)},
                              {**start, "stall_streak": float(n + 1)}, False, {})
               for n in range(steps))

    # (b) descend to floor 2, pick up a few items, then die halfway through
    tried = reward.compute(start, {**start, "depth": 2.0}, False, {})
    tried += reward.compute(start, {**start, "inventory_count": 7.0}, False, {})
    tried += reward.compute({**start, "depth": 2.0},
                            {**start, "depth": 2.0, "hp_current": 0.0}, True, {})

    assert tried > idle, "trying and dying must beat never leaving floor 1"


def test_wasted_action_is_penalized():
    from pilot.ml.rl.spd import spd_training_reward

    reward = spd_training_reward()
    prev = {"hp_current": 20.0, "depth": 1.0}
    effective = reward.compute(prev, {"hp_current": 20.0, "depth": 1.0, "action_wasted": 0}, False, {})
    wasted = reward.compute(prev, {"hp_current": 20.0, "depth": 1.0, "action_wasted": 1}, False, {})
    assert wasted < effective          # spamming an impossible no-op costs
    # the sim never emits the field -> no accidental penalty there
    assert reward.compute(prev, {"hp_current": 20.0, "depth": 1.0}, False, {}) == effective


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
