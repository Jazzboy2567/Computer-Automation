"""Deploy a trained policy onto the live Shattered Pixel Dungeon.

The screenshot -> feature-extract -> mouse-click seam for the real game. The
perception (`read_state`) and the screenshot/click functions are injected — at
runtime they are backed by the computer-use MCP; in tests they're fakes — so
this module is testable with no game and no desktop control.

Honest limits (the iterative real-game work):
* HUD reading (HP, the top-right enemy badge, depth, gold, inventory) is reliable
  at fixed positions. Positional cues the policy also wants (direction to nearest
  enemy / to the stairs) need on-map sprite detection; until that's added they
  default to "unknown", so the deployed policy is degraded vs the simulator.
* SPD's click-to-path UI doesn't map 1:1 to the sim's N/S/E/W steps — the camera
  centres on the hero, so a directional step is a click on the adjacent tile, but
  "descend" (walk onto the down-stairs) needs the stairs' on-screen position.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .agent import QLearningAgent
from .capture import ActionDriver, FeatureExtractor
from .env import Observation
from .spd_sim import SPDGridEnv, spd_featurizer

# A perceiver turns a screenshot + calibration into raw HUD readings.
ReadState = Callable[[Any, "SPDCalibration"], dict[str, float]]
ClickFn = Callable[[int, int], None]
ScreenshotFn = Callable[[], Any]


class SPDCalibration(BaseModel):
    """Window geometry for reading the HUD and computing click targets.

    All in screen pixels; calibrated once against the live game window (it
    depends on resolution / window size). Region boxes are (x, y, w, h).
    """

    board_center: tuple[int, int]              # the hero's on-screen pixel (camera-centred)
    tile_px: int                               # pixels per map tile (for directional clicks)
    heal_slot: tuple[int, int] = (0, 0)        # pixel of the heal/potion inventory slot
    hp_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    enemy_badge_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    depth_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    gold_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    inventory_region: tuple[int, int, int, int] = (0, 0, 0, 0)


def _not_calibrated(frame: Any, cal: SPDCalibration) -> dict[str, float]:
    raise NotImplementedError(
        "No perception wired. Provide a read_state(frame, calibration) that reads "
        "the HUD (HP X/Y, enemy badge, depth, gold, inventory) from the screenshot "
        "— calibrated against the live game window."
    )


class SPDFeatureExtractor(FeatureExtractor):
    """Screenshot -> SPD observation, via an injected HUD perceiver."""

    def __init__(self, read_state: ReadState = _not_calibrated, calibration: Optional[SPDCalibration] = None):
        self.read_state = read_state
        self.calibration = calibration

    def extract(self, frame: Any) -> Observation:
        raw = self.read_state(frame, self.calibration)
        hp = float(raw.get("hp_current", 0.0))
        hp_max = float(raw.get("hp_max", 0.0)) or 1.0
        hp_frac = max(0.0, min(1.0, hp / hp_max))
        return {
            # HUD fields (reliable)
            "hp_current": hp,
            "hp_max": hp_max,
            "hp_frac": hp_frac,
            "level": float(raw.get("level", 1.0)),
            "xp_frac": float(raw.get("xp_frac", 0.0)),
            "depth": float(raw.get("depth", 1.0)),
            "gold": float(raw.get("gold", 0.0)),
            "enemies_visible": float(raw.get("enemies_visible", 0.0)),
            "inventory_count": float(raw.get("inventory_count", 0.0)),
            "starving": float(raw.get("starving", 0.0)),
            # agent decision cues
            "hp_bin": float(min(4, int(hp_frac * 5))),
            "has_heal": float(raw.get("has_heal", 0.0)),
            # positional cues — need on-map CV; unknown for now (policy degrades)
            "enemy_dir": float(raw.get("enemy_dir", 0.0)),
            "enemy_adjacent": float(raw.get("enemy_adjacent", 0.0)),
            "stairs_dir": float(raw.get("stairs_dir", 0.0)),
            "stairs_dist": float(raw.get("stairs_dist", 0.0)),
        }


class SPDActionDriver(ActionDriver):
    """Map a policy action to a mouse click on the live game."""

    def __init__(self, click_fn: ClickFn, calibration: SPDCalibration):
        self.click = click_fn
        self.cal = calibration

    def do(self, action: str) -> None:
        cx, cy = self.cal.board_center
        t = self.cal.tile_px
        targets = {
            "move_n": (cx, cy - t),
            "move_s": (cx, cy + t),
            "move_e": (cx + t, cy),
            "move_w": (cx - t, cy),
        }
        if action in targets:
            self.click(*targets[action])
        elif action == "use_heal":
            self.click(*self.cal.heal_slot)
        elif action == "descend":
            # Walking onto the down-stairs auto-descends; without the stairs'
            # on-screen position we approximate by stepping toward them. Until
            # map-CV provides it, treat as a forward step.
            self.click(cx, cy - t)
        # "wait" / unknown: no click


def run_spd_live(
    q_table: dict,
    screenshot_fn: ScreenshotFn,
    click_fn: ClickFn,
    calibration: SPDCalibration,
    read_state: ReadState,
    max_turns: int = 200,
    settle: float = 0.4,
    done_fn: Optional[Callable[[Observation], bool]] = None,
    on_step: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """Run the trained policy on the live game; return a per-turn transcript.

    `screenshot_fn`/`click_fn` are backed by computer-use at runtime. The loop is
    turn-based, so it waits `settle` seconds after each click for the new turn.
    """
    agent = QLearningAgent(SPDGridEnv.action_space)
    agent.Q = dict(q_table)
    extractor = SPDFeatureExtractor(read_state, calibration)
    driver = SPDActionDriver(click_fn, calibration)
    done_fn = done_fn or (lambda o: o.get("hp_current", 1.0) <= 0)

    log: list[dict] = []
    for turn in range(max_turns):
        obs = extractor.extract(screenshot_fn())
        action = agent.policy(spd_featurizer(obs))
        entry = {
            "turn": turn,
            "action": action,
            "hp": obs["hp_current"],
            "depth": obs["depth"],
            "enemies_visible": obs["enemies_visible"],
        }
        log.append(entry)
        if on_step:
            on_step(entry)
        if done_fn(obs):
            break
        driver.do(action)
        if settle:
            time.sleep(settle)
    return log
