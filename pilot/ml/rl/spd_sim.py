"""A small SPD-like dungeon simulator for *training* (code-level, no screen).

Captures the Shattered Pixel Dungeon mechanics that matter for a policy —
fight approaching enemies, level up, heal at the right time, grab loot, descend
— sharing SPD's observation fields and the `spd_reward_spec()` reward. Train
here fast; deploy the learned policy onto the real game via the screenshot →
extract → mouse-click seam (`capture.ScreenGameEnv`).

It is an approximation (a sim-to-real gap exists); a faithful env would embed the
open-source SPD game itself.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .env import GameEnv, Observation

# (dx, dy) per move, with y increasing downward.
_MOVES = {"move_n": (0, -1), "move_s": (0, 1), "move_e": (1, 0), "move_w": (-1, 0)}
# 8-way direction code from a (sign dx, sign dy) pair; 0 = none/on-top.
_DIR8 = {
    (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4,
    (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8,
}


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _dir8(dx: int, dy: int) -> int:
    return _DIR8.get((_sign(dx), _sign(dy)), 0)


class SPDGridEnv(GameEnv):
    """Grid roguelike with SPD-flavored state, observation, and dynamics."""

    action_space = ["move_n", "move_s", "move_e", "move_w", "descend", "use_heal", "wait"]

    def __init__(self, seed: int = 0, size: int = 5, max_steps: int = 150):
        self._rng = random.Random(seed)
        self.size = size            # interior is 1..size on each axis
        self.max_steps = max_steps

    # ---------------------------------------------------------------- setup
    def _free_tile(self, taken: set[tuple[int, int]]) -> tuple[int, int]:
        while True:
            p = (self._rng.randint(1, self.size), self._rng.randint(1, self.size))
            if p not in taken:
                return p

    def _gen_floor(self) -> None:
        self.player = (1, 1)
        self.stairs = (self.size, self.size)
        taken = {self.player, self.stairs}
        self.enemies = []  # list of [x, y, hp]
        for _ in range(min(1 + self.depth, 4)):
            x, y = self._free_tile(taken)
            taken.add((x, y))
            self.enemies.append([x, y, 4 + self.depth])
        self.loot = self._free_tile(taken)

    def reset(self) -> Observation:
        self.depth = 1
        self.level = 1
        self.xp = 0
        self.hp_max = 20
        self.hp = 20
        self.gold = 0
        self.heals = 2
        self.items = 1            # non-heal items; inventory_count = heals + items
        self.hunger = 0
        self.steps = 0
        self._gen_floor()
        return self._obs()

    # ---------------------------------------------------------------- obs
    def observation_fields(self) -> list[str]:
        return ["hp_current", "hp_max", "hp_frac", "level", "xp_frac", "depth",
                "gold", "enemies_visible", "inventory_count", "starving"]

    def _nearest_enemy(self):
        if not self.enemies:
            return None
        px, py = self.player
        return min(self.enemies, key=lambda e: max(abs(e[0] - px), abs(e[1] - py)))

    def _obs(self) -> Observation:
        px, py = self.player
        ne = self._nearest_enemy()
        if ne:
            dx, dy = ne[0] - px, ne[1] - py
            enemy_dir = _dir8(dx, dy)
            enemy_adjacent = 1.0 if max(abs(dx), abs(dy)) == 1 else 0.0
        else:
            enemy_dir, enemy_adjacent = 0.0, 0.0
        sdx, sdy = self.stairs[0] - px, self.stairs[1] - py
        stairs_dist = max(abs(sdx), abs(sdy))
        starving = 1.0 if self.hunger > 60 else 0.0
        return {
            # full SPD fields (used by the reward)
            "hp_current": float(self.hp),
            "hp_max": float(self.hp_max),
            "hp_frac": self.hp / self.hp_max if self.hp_max else 0.0,
            "level": float(self.level),
            "xp_frac": self.xp / (10.0 * self.level),
            "depth": float(self.depth),
            "gold": float(self.gold),
            "enemies_visible": float(min(3, len(self.enemies))),
            "inventory_count": float(self.heals + self.items),
            "starving": starving,
            # compact decision cues (used by the agent's featurizer)
            "hp_bin": float(min(4, int(self.hp / self.hp_max * 5)) if self.hp_max else 0),
            "enemy_dir": float(enemy_dir),
            "enemy_adjacent": enemy_adjacent,
            "stairs_dir": float(_dir8(sdx, sdy)),
            "stairs_dist": float(stairs_dist),
            "has_heal": 1.0 if self.heals > 0 else 0.0,
        }

    # ---------------------------------------------------------------- step
    def _in_bounds(self, x: int, y: int) -> bool:
        return 1 <= x <= self.size and 1 <= y <= self.size

    def _enemy_at(self, x: int, y: int):
        for e in self.enemies:
            if e[0] == x and e[1] == y:
                return e
        return None

    def _level_up_check(self) -> None:
        while self.xp >= 10 * self.level:
            self.xp -= 10 * self.level
            self.level += 1
            self.hp_max += 5
            self.hp = self.hp_max  # level-up fully heals (SPD-like)

    def step(self, action: str) -> tuple[Observation, bool, dict[str, Any]]:
        self.steps += 1
        self.hunger += 1
        px, py = self.player

        if action in _MOVES:
            dx, dy = _MOVES[action]
            tx, ty = px + dx, py + dy
            if self._in_bounds(tx, ty):
                enemy = self._enemy_at(tx, ty)
                if enemy is not None:
                    enemy[2] -= 3 + self.level  # attack
                    if enemy[2] <= 0:
                        self.enemies.remove(enemy)
                        self.xp += 4
                        self.gold += self._rng.randint(0, 3)
                        self._level_up_check()
                else:
                    self.player = (tx, ty)
                    if self.player == self.loot:
                        self.items += 1
                        self.gold += self._rng.randint(1, 5)
                        self.loot = (-1, -1)
        elif action == "descend":
            if self.player == self.stairs:
                self.depth += 1
                self._gen_floor()
        elif action == "use_heal":
            if self.heals > 0 and self.hp < self.hp_max:
                self.hp = min(self.hp_max, self.hp + 15)
                self.heals -= 1
        # "wait" does nothing extra

        # Enemies act: attack if adjacent, else step toward the player.
        px, py = self.player
        for e in self.enemies:
            ex, ey = e[0], e[1]
            if max(abs(ex - px), abs(ey - py)) == 1:
                self.hp -= 3 + self.depth // 2
            else:
                nx, ny = ex + _sign(px - ex), ey + _sign(py - ey)
                if self._in_bounds(nx, ny) and self._enemy_at(nx, ny) is None and (nx, ny) != self.player:
                    e[0], e[1] = nx, ny

        # Pressure: more enemies keep arriving, so camping a floor isn't safe —
        # descending becomes the smart move (SPD-like: don't linger).
        if len(self.enemies) < 5 and self._rng.random() < 0.06:
            taken = {self.player, self.stairs, self.loot} | {(e[0], e[1]) for e in self.enemies}
            ex, ey = self._free_tile(taken)
            self.enemies.append([ex, ey, 4 + self.depth])

        # Slow passive regen when not starving.
        if self.hunger <= 60 and self.steps % 5 == 0:
            self.hp = min(self.hp_max, self.hp + 1)

        self.hp = max(0.0, self.hp)
        done = self.hp <= 0 or self.steps >= self.max_steps
        return self._obs(), done, {"depth": self.depth}


# Compact decision-state the tabular agent learns over (reward still uses the
# full observation). Keys are small ints so the Q-table stays tractable.
# Capability keys (has_food, wand_charges, gear_available, ...) make item/gear
# timing LEARNABLE — the env never decides when to use them. The gear-progression
# keys (str_potions, upgrade_scrolls, cursed_equipped, misc_available) let the
# agent learn spend-now vs bank-and-swing without any of it being scripted.
# Missing keys (the sim doesn't emit capability fields) default to 0.
_AGENT_KEYS = ("hp_bin", "enemies_visible", "enemy_dir", "enemy_adjacent",
               "stairs_dir", "has_heal", "starving",
               "has_food", "wand_charges", "gear_available", "challenge_count",
               "enemy_unaware", "has_missile", "loot_here", "has_bow",
               "has_unknown_potion", "has_unknown_scroll",
               "str_potions", "upgrade_scrolls", "cursed_equipped",
               "misc_available", "frontier_left", "loot_visible")

# Counts are capped so the table doesn't split hairs between large pools; the
# cap on upgrade_scrolls is generous (3) because "do I have a stack worth saving?"
# is exactly the distinction the agent needs for the bank-the-scrolls strategy.
_CAPS = {"wand_charges": 2.0, "str_potions": 2.0, "upgrade_scrolls": 3.0}


def spd_featurizer(obs: Observation) -> Observation:
    feat = {k: obs.get(k, 0.0) for k in _AGENT_KEYS}
    for k, cap in _CAPS.items():
        feat[k] = min(cap, feat[k])
    return feat


# The egocentric map window (rlbridge Observations.MAP_*): one bitmask per tile,
# 10 channels. The DQN needs it unpacked into 0/1 planes — a categorical bitmask
# fed raw would imply false ordinality between terrain types.
_MAP_CHANNELS = 10


def spd_map_featurizer(obs: Observation) -> Observation:
    """Identity over scalars, but unpack `map_bits` into a flat 0/1 plane vector
    under key `map` (kept as a numpy array so the DQN concatenates it directly).
    Used only by the neural agent — the tabular one can't take ~800 features."""
    feat = {k: v for k, v in obs.items() if k != "map_bits"}
    bits = obs.get("map_bits")
    if bits is not None:
        b = np.asarray(bits, dtype=np.int32)
        planes = ((b[:, None] >> np.arange(_MAP_CHANNELS)) & 1).astype(np.float32)
        feat["map"] = planes.ravel()
    return feat
