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

import os

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
    "enemies_killed": "cumulative kills this run (bridge: Statistics.enemiesSlain; monotonic, un-farmable) — the honest kill reward reads this",
    "inventory_count": "number of occupied inventory slots (moderate: count non-empty slots)",
    "starving": "1 if starving (no passive regen), else 0 (needs the hunger/buff icon or examine)",
    "has_ankh": "1 if an Ankh is held (revive item) (moderate: detect inventory item)",
    # Status effects (buffs/debuffs) are read via the magnifying-glass examine.
    # Per focused-enemy fields (enemy0..3_*) come from the bridge's focus block:
    # dx/dy/aware/hp/cooldown, plus:
    "enemyN_charging": "per focused enemy: telegraphed heavy-attack wind-up, player-visible (Goo 'pumping up': 1 = charging, 2 = about to release a doubled hit; 0 = none) — the agent LEARNS to react (step away / heal / burst), nothing scripted",
    "boss_hp_frac": "visible boss's HP fraction 0..1 (player reads the boss HP bar); 0 when no boss is in view — the reward scores damage dealt / healing during the boss fight",
    "boss_overheal": "1 after >2 consecutive turns of the boss regaining HP (derived in the env); flags the flee-and-let-it-heal stall (2-turn grace for a pump-up dodge)",
    "boss_pending": "1 from first sight of a boss until the hero gets past its floor (derived in the env); the reward taxes each such turn so ignoring the boss and running out the clock is not free (the 'want to battle' pressure)",
}

# Discrete actions. Capabilities only — WHEN to use each is the agent's to
# learn, never scripted (the project's no-hardcoding rule).
SPD_ACTIONS: list[str] = [
    "move_n", "move_s", "move_e", "move_w",   # step toward an adjacent tile
    "attack_nearest",                          # melee the nearest visible enemy
    # target SELECTION: melee the 2nd/3rd/4th nearest visible enemy (the same
    # enemy1/2/3 the observation lists) — kill-priority is the agent's to learn,
    # not hardcoded to always-nearest
    "attack_enemy_1", "attack_enemy_2", "attack_enemy_3",
    "search",                                  # magnifying glass (find hidden things / examine)
    "pickup",                                  # step onto / grab loot on the tile
    "descend",                                 # take the down-stairs (holding the Amulet: finish)
    "use_item",                                # drink a healing potion (poison under some challenges!)
    "explore",                                 # walk to the nearest unexplored area (tap the dark)
    "eat_food",                                # eat carried food
    "zap_wand",                                # cast a charged wand at the nearest visible enemy
    # zap a CHOSEN enemy (2nd/3rd/4th visible) — focus-fire the priority target
    "zap_enemy_1", "zap_enemy_2", "zap_enemy_3",
    # gear progression — FOUR separate decisions, never one macro, so the agent
    # (not the engine) learns the timing: spend an upgrade now vs bank six for a
    # big weapon, drink strength on find vs hold it, equip an upgrade vs a ring
    "equip_gear",                              # wear a strictly-better weapon/armor you have the STR for
    "equip_misc",                              # put on a ring/artifact if a trinket slot is free
    "drink_strength",                          # drink an identified Potion of Strength (+1 STR, permanent)
    # upgrade WEAPON vs ARMOR are two separate scroll decisions (weapon = damage now,
    # armor = damage-reduction to survive a boss) — the agent chooses, never scripted
    "read_upgrade",                            # read a Scroll of Upgrade into your WEAPON
    "read_upgrade_armor",                      # read a Scroll of Upgrade into your ARMOR
    "throw_item",                              # throw the best missile weapon at the nearest visible enemy
    "shoot_bow",                               # fire the Spirit Bow (huntress) at the nearest visible enemy
    "quaff_unknown",                           # drink an unidentified potion (the ID gamble)
    "read_unknown",                            # read an unidentified scroll (the ID gamble)
    # spend a talent point on the Nth perk of the current tier — the agent picks
    # the build (which perk, when); restores perks / rogue search / huntress vision
    "talent_0", "talent_1", "talent_2", "talent_3",
    "wait",                                    # pass a turn (passive regen)
]


def _hp_penalty_weight() -> float:
    try:
        return max(0.0, float(os.environ.get("PILOT_HP_PENALTY", "0") or "0"))
    except ValueError:
        return 0.0


_HP_PENALTY = _hp_penalty_weight()


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
            # Threat removal, done HONESTLY: reward actual KILLS via the cumulative
            # enemies_killed counter (Statistics.enemiesSlain), which only ever rises
            # — so it cannot be farmed. This REPLACES an old `enemies_visible down`
            # rule that fired whenever a foe left the hero's FIELD OF VIEW (not just on
            # a kill). A diagnostic (2026-08-03) found the greedy policy oscillating
            # enemies_visible ~73x/episode (down then up as an enemy re-entered view),
            # farming +1 each drop — ~99% of eval return, ~8x actual kills — loitering
            # by enemies (dying 45%) instead of descending. Removing it wholesale then
            # over-corrected: the reward went so sparse the agent COLLAPSED to a
            # passive wait basin (waited 40% of turns, stall_streak ~45, descended 28%)
            # because doing nothing dodged the HP/death penalties better than sparse
            # risky play earned. This honest kill reward restores a dense positive
            # signal for active play. It does NOT itself create descent pressure —
            # it pays for kills wherever they happen and gives no gradient toward the
            # stairs; descent pressure comes from the compounding depth term below and
            # the stall clock. (An earlier note here claimed kills "pull toward descent"
            # because a floor's foes are finite — that conflates "the floor runs out of
            # enemies" with a signal to descend, which it has none of. Stated honestly
            # to avoid the false-proxy thinking that produced the FOV rule.)
            RewardRule(field="enemies_killed", direction="up", weight=2.0, per_unit=True),
            # Descending COMPOUNDS (reaching floor d is worth ~5*d, so floor 5 far
            # outweighs floor 2 — a flat rate told the agent depth doesn't compound,
            # and it farmed floor 1 forever), but it only pays for a floor you
            # actually WORKED: the value is also scaled by how much of the floor you
            # left had been explored. Diving straight past a floor's loot is worth
            # almost nothing; taking the stairs after clearing it pays in full.
            # Together with the stall clock — which pushes you off a picked-clean
            # floor — this makes the intended loop optimal: loot the floor, then go
            # deeper, with each floor worth more than the last.
            RewardRule(field="depth", direction="up", weight=5.0,
                       scale_by=["depth", "descent_explored"]),
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
            # The ramp SATURATES (scale_cap): it climbs for ~25 stalled turns and
            # then holds. Uncapped it grew quadratically — a stuck greedy policy
            # bled about -125 a run (and would reach -578 over a full episode),
            # which both drowned the loot/depth signal and blew up the TD targets
            # the network fits. Capped, ~50 stalled turns still costs more than a
            # descent pays, which is the calibration that matters.
            RewardRule(field="stall_streak", direction="up", weight=-0.01,
                       per_unit=True, scale_by="stall_streak", scale_cap=25.0),
            RewardRule(field="has_amulet", direction="up", weight=200.0),                   # the Amulet of Yendor!
            RewardRule(field="won", direction="up", weight=500.0),                          # finishing the game = best
            # OPTIONAL convex HP penalty (env PILOT_HP_PENALTY, a float weight; 0/off
            # by default). Diagnosis of the floor-2 plateau: the agent fights every
            # enemy and trades HP for kills until it dies, because losing HP is nearly
            # free (-0.05/hp) next to a kill (+5 level). This adds a term that costs
            # MORE the lower the hero already is (scaled by hp_missing_frac, 0 at full
            # health, ~1 near death), so damage taken while low hurts a lot while a
            # scratch at full HP stays cheap. It teaches preserve-health / disengage /
            # heal as a LEARNED consequence — no scripted retreat. Runs A/B against
            # the baseline (var unset) to isolate whether valuing HP breaks the wall.
            *([RewardRule(field="hp_current", direction="down", per_unit=True,
                          weight=-_HP_PENALTY, scale_by="hp_missing_frac")]
              if _HP_PENALTY > 0 else []),
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
        # Reward raising STRENGTH. A diagnostic (2026-08-23) found the agent never
        # completes the gear-up chain even when handed the materials: it drinks a
        # Potion of Strength only sporadically and NEVER equips (equip_gear=0),
        # because drinking strength has no immediate payoff — STR itself wasn't
        # rewarded, so "drink -> drink -> THEN equip tier-2" is a delayed multi-step
        # reward whose first steps look like wasted turns. STR is pure upside and
        # monotonic (you can't lose it barring a cursed Ring of Might the agent
        # chose to wear; strength potions are finite), so rewarding its rise is
        # safe and un-farmable — like known_item_types. This makes the FIRST step of
        # the STR-unlock -> equip-tier2 chain rewarding so the whole chain becomes
        # learnable; equipping still pays via gear_score, so the agent learns the
        # sequence rather than being scripted into it.
        RewardRule(field="str", direction="up", weight=2.0, per_unit=True),
        # BOSS FIGHT (Matthew's lever, 2026-08-27). Diagnosis: the agent reaches Goo
        # but NEVER attacks it — Goo ends at full HP; it loots/mis-targets/dies, because
        # a ~100-HP boss is far too many hits for random exploration to ever KILL, so the
        # sparse +2 kill reward is never experienced and "attack the boss" is never
        # reinforced (0 beat-Goo even when handed tier-3 gear). Fix: make the boss's HP
        # BAR the objective during the fight. A big SYMMETRIC term pays per unit of damage
        # dealt (boss_hp_frac down) and un-pays for healing (up) — a dense per-hit signal
        # (~+0.8 a hit) that teaches attacking, while the damage<->heal cycle nets exactly
        # ZERO (un-farmable, the enemies_visible lesson) and telescopes to the real HP drop
        # so a legit 1-2 turn pump-up dodge isn't net-punished. Inert off the boss floor
        # (boss_hp_frac = 0 when no boss is visible) and on the sim (field absent -> 0).
        #
        # WEIGHT 16 (raised from 8, 2026-08-27). At weight 8 each hit paid ~+0.8 while
        # Goo's counter-hit cost ~-0.2..-1.8 (2x when pumped), so ATTACKING was roughly
        # break-even-to-negative per turn and the agent rationally AVOIDED the fight
        # (diagnostics: it paced/looted/no-op-spammed, Goo untouched). At weight 16 a hit
        # pays ~+1.6, clearly out-valuing the counter-hit, so attacking is net-POSITIVE
        # every turn — now 1-step TD can learn it greedily without needing to credit the
        # far-off kill. Still symmetric (heal costs the same) so it stays un-farmable.
        RewardRule(field="boss_hp_frac", direction="down", weight=16.0, per_unit=True),
        RewardRule(field="boss_hp_frac", direction="up", weight=-16.0, per_unit=True),
        # NOTE: a potential-based hp_frac shaping pair (reward healing, cost damage,
        # symmetrically) was tried here (2026-08-25) to stop the agent fighting
        # wounded — and did NOTHING: the death-diagnostic mean HP baseline stayed
        # ~13-15 and the death rate held at ~80%. Root cause: the agent has no
        # efficient way to heal early (slow passive regen, no potions), so it CAN'T
        # maintain a buffer no matter the incentive — it runs wounded because it is
        # UNDER-GEARED (takes damage faster than it can recover), not by choice.
        # Reverted. "Runs wounded" and "gets surrounded" are symptoms of the
        # materials wall, not independent, shapeable causes.
    ]
    # Per-turn cost for being SURROUNDED (2+ enemies adjacent). A death-cause graph
    # (2026-08-25) showed the agent lives perpetually wounded and dies to melee
    # attrition — 40% of deaths while surrounded — not to starvation or DoT. This
    # charges a small cost each turn it lets itself be flanked, teaching it to fight
    # from a chokepoint (it already sees the walkable ring) instead of in the open.
    # A STATE penalty, not a scripted retreat: the agent learns HOW to un-flank. Kept
    # modest so a genuinely worthwhile 1v2 (a finishing blow) can still be chosen.
    # ...plus the "don't let the boss heal" cost Matthew asked for: after MORE THAN
    # 2 straight turns of the boss regaining HP (a 2-turn grace to dodge a telegraphed
    # pump-up), each further healing turn is charged — punishing the flee-and-stall that
    # lets Goo heal back up, on top of the symmetric heal term above.
    #
    # NOTE: a boss_pending "want to battle" TAX (-0.1/turn while a met boss is unbeaten,
    # 2026-08-27) was tried to stop the avoidance and REVERTED — it backfired. Because
    # the agent can't yet WIN the fight, taxing "met the boss and haven't beaten it" just
    # taught it to avoid the boss FLOOR entirely (reachGoo collapsed to 0%, and forced
    # onto floor 5 it switched to a new no-op, zap_enemy_2 95%). You can't price-pressure
    # an agent into a fight it doesn't know how to win. The fix instead is above: make
    # each ATTACK immediately pay (weight 16) so engaging is the greedy choice. The
    # boss_pending FEATURE stays in the observation (harmless, tells the agent it's on a
    # live boss floor); only its penalty was removed.
    flags = {**base.flag_penalties, "surrounded": -0.15, "boss_overheal": -0.5}
    return base.model_copy(update={"rules": base.rules + shaping, "flag_penalties": flags})
