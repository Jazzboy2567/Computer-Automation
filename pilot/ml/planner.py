"""Planners — the BACKGROUND role for the LLM.

A planner turns a natural-language goal + a data profile into a structured
`MLTaskSpec`. That is the *only* job the LLM has here; it never produces the
user's result. `HeuristicPlanner` needs no AI at all (so ML always runs), and
`OllamaPlanner` uses a local model to better interpret fuzzy goals.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from .spec import MLTaskSpec

_CLUSTER_WORDS = ("cluster", "segment", "group", "unsupervised", "partition")
_REGRESS_WORDS = ("regress", "predict the value", "estimate", "forecast", "how much", "price")


def _profile_targets(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return profile.get("columns", [])


def _is_categorical_col(col: dict[str, Any], n_rows: int) -> bool:
    if not col.get("is_numeric", False):
        return True
    n = col.get("n_unique", 0)
    return n <= max(2, min(20, int(0.05 * n_rows) or 20))


class Planner(ABC):
    name = "base"

    @abstractmethod
    def plan(
        self,
        goal: str,
        profile: dict[str, Any],
        target_hint: Optional[str] = None,
    ) -> MLTaskSpec:
        raise NotImplementedError


class HeuristicPlanner(Planner):
    """Infer the task from the data + goal text — no model required."""

    name = "heuristic"

    def plan(self, goal: str, profile: dict[str, Any], target_hint: Optional[str] = None) -> MLTaskSpec:
        g = goal.lower()
        cols = _profile_targets(profile)
        names = [c["name"] for c in cols]
        n_rows = profile.get("n_rows", 0)

        # 1) explicit clustering ask, or no usable columns to predict
        wants_cluster = any(w in g for w in _CLUSTER_WORDS)

        # 2) pick a target: hint -> a column named in the goal -> hinted name -> last col
        target = target_hint
        if not target:
            for name in names:
                if str(name).lower() in g:
                    target = name
                    break
        if not target:
            from .datasets import _TARGET_HINTS

            lowered = {str(n).lower(): n for n in names}
            for h in _TARGET_HINTS:
                if h in lowered:
                    target = lowered[h]
                    break
        if not target and names:
            target = names[-1]

        if wants_cluster or target is None:
            k = None
            return MLTaskSpec(
                task_type="clustering",
                target=None,
                model="kmeans",
                metric="silhouette",
                n_clusters=k,
                planner=self.name,
                notes="goal/data implied unsupervised grouping",
            )

        target_col = next((c for c in cols if c["name"] == target), None)
        categorical = _is_categorical_col(target_col, n_rows) if target_col else True
        if any(w in g for w in _REGRESS_WORDS) and target_col and target_col.get("is_numeric"):
            categorical = False

        if categorical:
            return MLTaskSpec(
                task_type="classification", target=target, model="auto",
                metric="accuracy", planner=self.name,
                notes=f"target {target!r} looks categorical",
            )
        return MLTaskSpec(
            task_type="regression", target=target, model="auto",
            metric="r2", planner=self.name,
            notes=f"target {target!r} looks continuous",
        )


class OllamaPlanner(Planner):
    """Use a local Ollama model to choose the task spec (background role)."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None, timeout: float = 60.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def plan(self, goal: str, profile: dict[str, Any], target_hint: Optional[str] = None) -> MLTaskSpec:
        import httpx

        system = (
            "You are an ML task planner. Given a user goal and a dataset profile, "
            "choose how to solve it with classical ML. Respond with ONLY a JSON object "
            "with keys: task_type (classification|regression|clustering), target (column "
            "name or null), features (list, empty for all), model (auto|random_forest|"
            "linear|logistic|kmeans), metric, test_size (0-1), n_clusters (int or null), "
            "notes. Choose the target from the dataset columns."
        )
        cols = ", ".join(f"{c['name']}({c['dtype']},u={c['n_unique']})" for c in profile.get("columns", []))
        user = (
            f"GOAL: {goal}\n"
            f"DATASET: {profile.get('n_rows')} rows x {profile.get('n_cols')} cols\n"
            f"COLUMNS: {cols}\n"
            + (f"TARGET HINT: {target_hint}\n" if target_hint else "")
            + "Return the JSON task spec."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
                content = (resp.json().get("message") or {}).get("content", "")
                data = json.loads(content)
        except Exception as e:
            raise RuntimeError(
                f"Ollama planner failed ({type(e).__name__}: {e}). Is Ollama running "
                f"with model {self.model!r}? See https://ollama.com"
            ) from e
        data.setdefault("planner", self.name)
        if target_hint and not data.get("target"):
            data["target"] = target_hint
        return MLTaskSpec.model_validate(data)


class AutoPlanner(Planner):
    """Prefer the local LLM; fall back to the heuristic if it's unreachable."""

    name = "auto"

    def __init__(self):
        self._ollama = OllamaPlanner()
        self._heuristic = HeuristicPlanner()

    def plan(self, goal: str, profile: dict[str, Any], target_hint: Optional[str] = None) -> MLTaskSpec:
        try:
            return self._ollama.plan(goal, profile, target_hint)
        except Exception:
            return self._heuristic.plan(goal, profile, target_hint)


def get_planner(name: str = "auto") -> Planner:
    name = (name or "auto").lower()
    if name == "heuristic":
        return HeuristicPlanner()
    if name == "ollama":
        return OllamaPlanner()
    if name == "auto":
        return AutoPlanner()
    raise ValueError(f"Unknown planner: {name!r}")
