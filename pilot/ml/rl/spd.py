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
    # spend a talent point on the Nth perk of the current tier — the agent picks
    # the build (which perk, when); restores perks / rogue search / huntress vision
    "talent_0", "talent_1", "talent_2", "talent_3",
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
            # Descending COMPOUNDS: reaching floor d pays 5*d, so floor 5 (+25) is
            # worth far more than floor 2 (+10). A flat rate told the agent that
            # depth doesn't compound — and since fully exploring a floor paid more
            # than the stairs, the optimal policy was to farm floor 1 and never
            # descend. Depth compounding IS the game (the Amulet is on floor 26).
            RewardRule(field="depth", direction="up", weight=5.0, scale_by="depth"),
            RewardRule(field="gold", direction="up", weight=0.01, per_unit=True),           # gold = good
            # LOOT is the objective, and it pays PER ITEM. Exploring a floor is not
            # rewarded in itself — it's simply how you find items, so it happens
            # because it pays off, not because walking is scored. Self-limiting: a
            # floor holds finitely many items, so once it's picked clean the only
            # way to get more loot is to go deeper.
            RewardRule(field="inventory_count", direction="up", weight=2.0, per_unit=True),
            # Getting STUCK costs progressively more. stall_streak counts turns that
            # achieved nothing — no fighting (or recent enemy), no healing, no new
            # ground, no loot, no floor change — and this charges 0.01 * streak each
            # such turn, so the cost ramps instead of firing at some magic cutoff.
            # Consequence: once a floor is picked clean and quiet, standing around
            # (or pacing it) bleeds reward until the hero takes the stairs, while a
            # fight, a rest that actually heals, or any real find resets it to free.
            # This is what stops "linger safely on floor 1" from beating a descent.
            RewardRule(field="stall_streak", direction="up", weight=-0.01,
                       per_unit=True, scale_by="stall_streak"),
            RewardRule(field="has_amulet", direction="up", weight=200.0),                   # the Amulet of Yendor!
            RewardRule(field="won", direction="up", weight=500.0),                          # finishing the game = best
        ],
        # No blanket per-turn cost: a flat charge would punish resting off damage,
        # which is real play (HP regen). Wasteful idling is charged instead — see
        # flag_penalties below.
        step_reward=0.0,
        flag_penalties={
            # An impossible action degraded to a rest: stops a policy hiding by
            # spamming a no-op (one DQN run collapsed onto descend-with-no-stairs).
            "action_wasted": -0.05,
        },
        death_field="hp_current",
        death_threshold=0.0,
        # Death is still the worst single event, but -50 against a +10 descent
        # made never playing the optimal policy: a descent needed >70% survival
        # odds just to break even, and an agent still learning to fight is far
        # below that, so it correctly refused to ever take the stairs. At -20 the
        # expected value of TRYING a floor beats idling the clock out, which is
        # what lets it practise depth and get good enough to survive there.
        death_reward=-20.0,
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
        # NOTE: covering ground is deliberately NOT rewarded. Paying per explored
        # cell made "walk every tile of floor 1" the optimal policy (+15-20 risk-free
        # vs a flat +10 descent) and the agent farmed the first floor for 150k
        # episodes. Exploring is the MEANS, not the end: the agent explores because
        # that is how it finds items and the stairs, both of which pay. Rewarding
        # the walking rewards a proxy for the goal instead of the goal.
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
