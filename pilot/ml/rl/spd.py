"""Shattered Pixel Dungeon: observation schema, action space, and reward.

This encodes the *game-specific* configuration for SPD — independent of whether
we drive it by screen-scraping (observe/assist) or a code-level environment
(real training). The reward below is a faithful translation of the user's
good/bad events; nuanced policy (when to heal, descend only after looting,
don't waste a full-heal potion, prefer higher-tier gear) is intentionally left
for the agent to *learn* from these outcome rewards rather than hardcoded.

SPD is turn-based, so the loop is unhurried: screenshot -> extract these fields
-> choose one action -> act -> screenshot the next turn.
"""

from __future__ import annotations

from .reward import RewardRule, RewardSpec

# The structured observation read from each frame (the "consistent important
# data"). Reliability notes mark how easily each is read from a screenshot.
SPD_OBSERVATION_FIELDS: dict[str, str] = {
    "hp_current": "current HP, from the bottom-left 'X/Y' bar (reliable: fixed-position text)",
    "hp_max": "max HP, the 'Y' in 'X/Y' (reliable)",
    "hp_frac": "hp_current / hp_max, 0..1 (derived)",
    "level": "hero level, bottom-left 'Lv. N' (reliable)",
    "xp_frac": "XP progress to next level, the '0/10' bar (reliable)",
    "depth": "current dungeon floor (reliable: 'descend to floor N' / depth indicator)",
    "gold": "gold held, inventory header (reliable: fixed-position number)",
    "enemies_visible": "count from the top-right 'N + skull' badge (reliable when present, else 0)",
    "inventory_count": "number of occupied inventory slots (moderate: count non-empty slots)",
    "starving": "1 if starving (no passive regen), else 0 (needs the hunger/buff icon or examine)",
    "has_ankh": "1 if an Ankh is held (revive item) (moderate: detect inventory item)",
    # Status effects (buffs/debuffs) are read via the magnifying-glass examine.
}

# Discrete actions, performed as mouse clicks via the ActionDriver (turn-based).
SPD_ACTIONS: list[str] = [
    "move_n", "move_s", "move_e", "move_w",   # step toward an adjacent tile
    "attack_nearest",                          # click the nearest visible enemy
    "search",                                  # magnifying glass (find hidden things / examine)
    "pickup",                                  # step onto / grab loot on the tile
    "descend",                                 # take the down-stairs
    "use_item",                                # use the best consumable (e.g. heal when low)
    "wait",                                    # pass a turn (passive regen)
]


def spd_reward_spec() -> RewardSpec:
    """The user's good/bad events for SPD, as reward over observation changes.

    Good: surviving, killing enemies (xp/level up), removing threats, descending,
    gaining gold/items. Bad: taking damage. Worst: death. Resource *waste* (e.g.
    healing at near-full) is deliberately NOT penalized directly — it shows up as
    future death risk, so the agent must learn good timing.
    """
    return RewardSpec(
        rules=[
            RewardRule(field="hp_current", direction="down", weight=-0.05, per_unit=True),  # taking damage = bad
            RewardRule(field="level", direction="up", weight=5.0),                          # level up (kills) = good
            RewardRule(field="xp_frac", direction="up", weight=2.0, per_unit=True),         # progress to a kill/level
            RewardRule(field="enemies_visible", direction="down", weight=1.0),              # threat removed = good
            RewardRule(field="depth", direction="up", weight=3.0),                          # descending = good
            RewardRule(field="gold", direction="up", weight=0.01, per_unit=True),           # gold = good
            RewardRule(field="inventory_count", direction="up", weight=1.0),                # more items = good
        ],
        step_reward=0.1,            # passive survival / regen when not starving
        death_field="hp_current",
        death_threshold=0.0,
        death_reward=-50.0,         # death = worst (an Ankh would soften this — future work)
    )
