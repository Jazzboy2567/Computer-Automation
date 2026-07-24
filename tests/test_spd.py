"""Tests for the Shattered Pixel Dungeon reward spec + schema (no game needed)."""

from __future__ import annotations

from pilot.ml.rl.spd import SPD_ACTIONS, SPD_OBSERVATION_FIELDS, spd_reward_spec


def test_schema_present():
    assert "hp_current" in SPD_OBSERVATION_FIELDS
    assert "enemies_visible" in SPD_OBSERVATION_FIELDS
    assert "attack_nearest" in SPD_ACTIONS and "descend" in SPD_ACTIONS


def test_reward_encodes_good_and_bad():
    spec = spd_reward_spec()

    base = {"hp_current": 20, "level": 1, "xp_frac": 0.0, "enemies_visible": 1,
            "depth": 1, "gold": 0, "inventory_count": 5}

    def obs(**changes):
        o = dict(base)
        o.update(changes)
        return o

    # taking damage is bad
    assert spec.compute(base, obs(hp_current=10), False, {}) < 0
    # killing -> level up is good (and large)
    assert spec.compute(base, obs(level=2), False, {}) > 4
    # descending is good
    assert spec.compute(base, obs(depth=2), False, {}) > 0
    # removing a threat is good
    assert spec.compute(base, obs(enemies_visible=0), False, {}) > 0
    # gaining an item is good
    assert spec.compute(base, obs(inventory_count=6), False, {}) > 0
    # death is the worst single event — asserted RELATIVE to the other events
    # rather than against a magic constant, so retuning the penalty (it was
    # lowered from -50 because paralysing risk-aversion made never descending
    # optimal) can't silently make death cheap
    death = spec.compute(obs(hp_current=5), obs(hp_current=0), True, {})
    assert death < 0
    for good in (obs(level=2), obs(depth=2), obs(enemies_visible=0), obs(inventory_count=6)):
        assert death < -spec.compute(base, good, False, {})
