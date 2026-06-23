"""SPD-like training simulator tests — offline, deterministic."""

from __future__ import annotations

from pathlib import Path

from pilot.ml.rl.spd_sim import SPDGridEnv, spd_featurizer
from pilot.ml.rl.spd_train import run_spd_training

# Approximate cardinal step toward an 8-way direction code (no diagonal moves).
_DIR_MOVE = {1: "move_n", 2: "move_e", 3: "move_e", 4: "move_e",
             5: "move_s", 6: "move_w", 7: "move_w", 8: "move_n"}


def test_obs_has_spd_fields():
    env = SPDGridEnv(seed=0)
    o = env.reset()
    for f in ("hp_current", "hp_max", "depth", "enemies_visible", "inventory_count",
              "stairs_dir", "stairs_dist", "has_heal"):
        assert f in o
    assert set(spd_featurizer(o)).issubset(set(o))  # featurizer is a subset of obs


def test_env_supports_descent():
    # A deliberate stairs-seeking policy can actually descend (env mechanics work).
    env = SPDGridEnv(seed=3)
    obs = env.reset()
    done, steps = False, 0
    while not done and steps < 200:
        sd = int(obs["stairs_dir"])
        action = "descend" if sd == 0 else _DIR_MOVE[sd]
        obs, done, info = env.step(action)
        steps += 1
    assert env.depth >= 2


def test_training_learns(tmp_path):
    result, ws = run_spd_training(episodes=6000, base_dir=tmp_path / "spd", seed=0)
    # Learns to survive far better than random...
    assert result.avg_return_trained > result.avg_return_random
    assert result.improvement > 2.0
    assert result.avg_survival_trained > result.avg_survival_random
    # ...and progresses at least as deep (it descends; random rarely does).
    assert result.avg_depth_trained >= result.avg_depth_random
    assert result.states_learned > 0
    assert Path(result.model_path).exists()
    assert (ws.path / "report.md").exists()
    assert (ws.path / "metrics.json").exists()
