"""SPD deployment-layer tests — fakes for perception + clicks (no game/MCP)."""

from __future__ import annotations

from pilot.ml.rl.spd_deploy import (
    SPDActionDriver, SPDCalibration, SPDFeatureExtractor, run_spd_live,
)
from pilot.ml.rl.spd_sim import SPDGridEnv


def test_extractor_builds_observation():
    def fake_read(frame, cal):
        return {"hp_current": 14, "hp_max": 20, "depth": 2, "gold": 5,
                "enemies_visible": 1, "inventory_count": 3, "has_heal": 1}

    obs = SPDFeatureExtractor(fake_read).extract("frame")
    assert obs["hp_current"] == 14 and obs["hp_max"] == 20
    assert abs(obs["hp_frac"] - 0.7) < 1e-6
    assert obs["hp_bin"] == 3                      # 0.7 -> bin 3
    assert obs["depth"] == 2 and obs["enemies_visible"] == 1 and obs["has_heal"] == 1
    # positional cues are present but unknown until map-CV is added
    assert obs["stairs_dir"] == 0 and obs["enemy_dir"] == 0


def test_action_driver_click_targets():
    clicks = []
    cal = SPDCalibration(board_center=(1000, 500), tile_px=40, heal_slot=(120, 880))
    drv = SPDActionDriver(lambda x, y: clicks.append((x, y)), cal)
    drv.do("move_n"); drv.do("move_e"); drv.do("use_heal"); drv.do("wait")
    assert clicks == [(1000, 460), (1040, 500), (120, 880)]  # 'wait' clicks nothing


def test_run_spd_live_loop():
    calls = {"n": 0}

    def fake_read(frame, cal):
        calls["n"] += 1
        hp = 20 if calls["n"] < 4 else 0          # "die" on the 4th frame
        return {"hp_current": hp, "hp_max": 20, "depth": 1}

    clicks = []
    cal = SPDCalibration(board_center=(800, 400), tile_px=32)
    log = run_spd_live(
        q_table={}, screenshot_fn=lambda: "frame",
        click_fn=lambda x, y: clicks.append((x, y)),
        calibration=cal, read_state=fake_read, max_turns=20, settle=0.0,
    )
    assert len(log) == 4               # screenshot/extract/decide each turn
    assert log[-1]["hp"] == 0          # loop ended when HP hit 0
    assert all(e["action"] in SPDGridEnv.action_space for e in log)
    assert 0 <= len(clicks) <= 3       # acted only on the 3 living turns
