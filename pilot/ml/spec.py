"""Schemas for ML goals: the task spec the planner produces and the result the
engine produces.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

TaskType = Literal["classification", "regression", "clustering"]


class MLTaskSpec(BaseModel):
    """A structured machine-learning task — the output of the background planner.

    This is the *only* thing the LLM produces. Everything downstream (training,
    evaluation, the result) is plain ML code with no model-in-the-loop.
    """

    task_type: TaskType
    target: Optional[str] = None          # column to predict (None for clustering)
    features: list[str] = Field(default_factory=list)  # empty => all but target
    model: str = "auto"                   # auto | random_forest | linear | logistic | kmeans
    metric: Optional[str] = None          # primary metric name (informational)
    test_size: float = 0.25
    n_clusters: Optional[int] = None      # clustering only
    planner: str = "heuristic"            # which planner produced this spec
    notes: Optional[str] = None           # planner reasoning / rationale

    def short(self) -> str:
        bits = [self.task_type]
        if self.target:
            bits.append(f"target={self.target}")
        bits.append(f"model={self.model}")
        return " ".join(bits)


class MLResult(BaseModel):
    """The engine's output — the user's actual result, plus how well it performed."""

    task_type: TaskType
    model: str
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: Optional[str] = None
    primary_score: Optional[float] = None
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    feature_importances: dict[str, float] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    model_path: Optional[str] = None
    predictions_path: Optional[str] = None

    def headline(self) -> str:
        if self.primary_metric and self.primary_score is not None:
            return f"{self.primary_metric}={self.primary_score:.4f}"
        return "trained"
