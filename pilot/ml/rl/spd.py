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

# Discrete actions. Capabilities only — WHEN to use each is the agent's to
# learn, never scripted (the project's no-hardcoding rule).
SPD_ACTIONS: list[str] = [
    "move_n", "move_s", "move_e", "move_w",   # step toward an adjacent tile
    "attack_nearest",                          # melee the nearest visible enemy
    "search",                                  # magnifying glass (find hidden things / examine)
    "pickup",                                  # step onto / grab loot on the tile
    "descend",                                 # take the down-stairs (holding the Amulet: finish)
    "use_item",                                # drink a healing potion (poison under some challenges!)
    "explore",                                 # walk to the nearest unexplored area (tap the dark)
    "eat_food",                                # eat carried food
    "zap_wand",                                # cast a charged wand at the nearest visible enemy
    # gear progression — FOUR separate decisions, never one macro, so the agent
    # (not the engine) learns the timing: spend an upgrade now vs bank six for a
    # big weapon, drink strength on find vs hold it, equip an upgrade vs a ring
    "equip_gear",                              # wear a strictly-better weapon/armor you have the STR for
    "equip_misc",                              # put on a ring/artifact if a trinket slot is free
    "drink_strength",                          # drink an identified Potion of Strength (+1 STR, permanent)
    "read_upgrade",                            # read an identified Scroll of Upgrade (targets your weapon)
    "throw_item",                              # throw the best missile weapon at the nearest visible enemy
    "shoot_bow",                               # fire the Spirit Bow (huntress) at the nearest visible enemy
    "quaff_unknown",                           # drink an unidentified potion (the ID gamble)
    "read_unknown",                            # read an unidentified scroll (the ID gamble)
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
            RewardRule(field="depth", direction="up", weight=10.0),                         # descending = strong progress
            RewardRule(field="gold", direction="up", weight=0.01, per_unit=True),           # gold = good
            RewardRule(field="inventory_count", direction="up", weight=1.0),                # more items = good
            RewardRule(field="has_amulet", direction="up", weight=200.0),                   # the Amulet of Yendor!
            RewardRule(field="won", direction="up", weight=500.0),                          # finishing the game = best
        ],
        # No flat survival bonus. A +0.05/step bonus quietly paid the agent to
        # STALL — surviving ~490 turns on floor 1 banked ~+24 and beat the risk
        # of a descent that might end in the death penalty, so a DQN run collapsed
        # to a do-nothing policy. Survival is now incentivised only by NOT dying
        # (the death penalty) and by staying alive to earn more PROGRESS reward.
        step_reward=0.0,
        death_field="hp_current",
        death_threshold=0.0,
        death_reward=-50.0,         # death = worst (an Ankh would soften this — future work)
    )


def spd_training_reward() -> RewardSpec:
    """The user's reward plus potential-based shaping toward the down-stairs.

    Descending is a sparse reward the agent rarely discovers by random
    exploration, so for *training* we add a symmetric (un-farmable) shaping term
    on distance-to-stairs — progress toward the (good) descend. Evaluation still
    uses the pure `spd_reward_spec()` so performance reflects the real objective.
    """
    base = spd_reward_spec()
    shaping = [
        RewardRule(field="stairs_dist", direction="down", weight=0.3, per_unit=True),
        RewardRule(field="stairs_dist", direction="up", weight=-0.3, per_unit=True),
        # Seeing new floor is progress ("check the floor for loot" / find the
        # stairs). Un-farmable: cells_explored only ever grows. The sim doesn't
        # emit the field, so this rule is inert there (missing field = 0 delta).
        RewardRule(field="cells_explored", direction="up", weight=0.05, per_unit=True),
        # Information gain: identifying an item TYPE (drink an unknown potion,
        # read an unknown scroll, zap an unknown wand, wear a ring) pays out, so
        # the agent learns the ID gamble is worth the risk instead of hoarding
        # mystery items it can never knowingly use. The poison/curse/death costs
        # stay in the base reward, so it must LEARN which gambles are worth it —
        # nothing here scripts what to identify or when. Un-farmable: the count
        # only rises within a run and resets each game.
        RewardRule(field="known_item_types", direction="up", weight=3.0, per_unit=True),
        # A SLIGHT nudge to TRY better gear early: pays when equipped weapon/armor
        # tier or (known) upgrade level rises — a real improvement, not a lateral
        # swap. Gated to floors 1-10 (Matthew: once a build is going, switching
        # gear isn't the best), and small enough that survival/depth still lead.
        # The agent still learns WHICH gear and WHEN; this only makes trying it on
        # worth the detour now that stalling earns nothing.
        RewardRule(field="gear_score", direction="up", weight=1.5, per_unit=True, max_depth=10),
    ]
    return base.model_copy(update={"rules": base.rules + shaping})
