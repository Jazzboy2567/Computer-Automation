"""ML foreground.

In this subsystem the *machine-learning engine* is the thing that produces the
user's result, and the LLM (Ollama) is demoted to a **background planner** that
only translates a natural-language goal into a structured task spec. The LLM is
optional — a no-AI `HeuristicPlanner` infers the task from the data so ML always
runs.

Flow (see ``pilot.ml.orchestrator``):

    goal text ─▶ planner (Ollama or heuristic) ─▶ MLTaskSpec
                                                       │
    data ─────────────────────────────────────────────┼─▶ engine (scikit-learn)
                                                       ▼
                              isolated workspace  ◀── result + metrics + report

Each goal runs in its own isolated `MLWorkspace`.
"""

from __future__ import annotations

from .spec import MLResult, MLTaskSpec
from .workspace import MLWorkspace

__all__ = ["MLTaskSpec", "MLResult", "MLWorkspace"]
