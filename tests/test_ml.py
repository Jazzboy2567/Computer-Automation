"""ML-foreground tests — fully offline (heuristic planner, no Ollama, no network).

Covers workspace isolation, heuristic planning, and the classification /
regression / clustering engines end-to-end through the orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.ml import datasets
from pilot.ml.orchestrator import run_ml_goal
from pilot.ml.planner import HeuristicPlanner
from pilot.ml.workspace import MLWorkspace


def _classification_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    label = ((x1 + x2) > 0).astype(int)  # learnable boundary
    df = pd.DataFrame({"x1": x1, "x2": x2, "label": label})
    p = tmp_path / "clf.csv"
    df.to_csv(p, index=False)
    return p


def _regression_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    noise = rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"feature": x, "value": 3.0 * x + 2.0 + noise})
    p = tmp_path / "reg.csv"
    df.to_csv(p, index=False)
    return p


# ----------------------------------------------------------------- workspace
def test_workspace_isolation(tmp_path):
    a = MLWorkspace.create("predict churn", base_dir=tmp_path)
    b = MLWorkspace.create("predict churn", base_dir=tmp_path)
    assert a.path != b.path  # separate workspace per goal/run
    for ws in (a, b):
        assert ws.data_dir.exists() and ws.model_dir.exists() and ws.results_dir.exists()
        assert (ws.path / "goal.json").exists()


# ----------------------------------------------------------------- planner
def test_heuristic_planner_picks_task():
    clf_profile = {"n_rows": 100, "n_cols": 3, "columns": [
        {"name": "x1", "dtype": "float64", "n_unique": 100, "is_numeric": True},
        {"name": "x2", "dtype": "float64", "n_unique": 100, "is_numeric": True},
        {"name": "label", "dtype": "int64", "n_unique": 2, "is_numeric": True},
    ]}
    spec = HeuristicPlanner().plan("predict the label", clf_profile)
    assert spec.task_type == "classification" and spec.target == "label"

    reg_profile = {"n_rows": 100, "n_cols": 2, "columns": [
        {"name": "feature", "dtype": "float64", "n_unique": 100, "is_numeric": True},
        {"name": "value", "dtype": "float64", "n_unique": 100, "is_numeric": True},
    ]}
    spec = HeuristicPlanner().plan("estimate the value", reg_profile, target_hint="value")
    assert spec.task_type == "regression" and spec.target == "value"

    spec = HeuristicPlanner().plan("segment customers into groups", reg_profile)
    assert spec.task_type == "clustering"


# ----------------------------------------------------------------- engines
def test_classification_end_to_end(tmp_path):
    csv = _classification_csv(tmp_path)
    result, ws = run_ml_goal(
        "classify the label", data_path=str(csv), target="label",
        planner="heuristic", base_dir=tmp_path / "ws",
    )
    assert result.task_type == "classification"
    assert result.primary_metric == "accuracy"
    assert result.primary_score > 0.8  # learnable boundary
    assert Path(result.model_path).exists()
    assert (ws.path / "report.md").exists()
    metrics = json.loads((ws.path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["accuracy"] == result.metrics["accuracy"]


def test_regression_end_to_end(tmp_path):
    csv = _regression_csv(tmp_path)
    result, ws = run_ml_goal(
        "predict the value", data_path=str(csv), target="value",
        planner="heuristic", base_dir=tmp_path / "ws",
    )
    assert result.task_type == "regression"
    assert result.primary_score > 0.9  # near-linear relationship
    assert "rmse" in result.metrics
    assert Path(result.predictions_path).exists()


def test_clustering_end_to_end(tmp_path):
    csv = _classification_csv(tmp_path)  # 2 numeric features
    result, ws = run_ml_goal(
        "cluster the rows into groups", data_path=str(csv),
        planner="heuristic", base_dir=tmp_path / "ws",
    )
    assert result.task_type == "clustering"
    assert result.metrics["n_clusters"] >= 2
    assert result.extra["cluster_sizes"]


def test_bundled_sample_runs_with_no_data(tmp_path):
    # No data + no Ollama: heuristic planner + bundled iris still produces a result.
    result, ws = run_ml_goal(
        "classify iris species", planner="heuristic", base_dir=tmp_path / "ws",
    )
    assert result.task_type == "classification"
    assert result.primary_score > 0.8
    assert (ws.data_dir / "dataset.csv").exists()  # snapshot of data used
