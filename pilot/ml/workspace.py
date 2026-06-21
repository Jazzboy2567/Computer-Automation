"""Per-goal isolated ML workspaces.

Every ML goal gets its own directory under ``ml_workspaces/`` holding its data,
model artifact, results and a report — so goals never tread on each other.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import ML_WORKSPACES_DIR


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48] or "goal"


class MLWorkspace:
    """An isolated folder for one ML goal: ``<base>/<slug>_<timestamp>/``."""

    def __init__(self, path: Path, goal: str):
        self.path = path
        self.goal = goal
        self.data_dir = path / "data"
        self.model_dir = path / "model"
        self.results_dir = path / "results"
        for d in (self.data_dir, self.model_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, goal: str, base_dir: Optional[Path] = None) -> "MLWorkspace":
        base = base_dir or ML_WORKSPACES_DIR
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = base / f"{_slug(goal)}_{ts}"
        path.mkdir(parents=True, exist_ok=True)
        ws = cls(path, goal)
        ws.write_json("goal.json", {"goal": goal, "created": datetime.now().isoformat(timespec="seconds")})
        return ws

    # ------------------------------------------------------------------ io
    def write_json(self, name: str, data: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path / name
        p.write_text(text, encoding="utf-8")
        return p

    def read_json(self, name: str) -> Any:
        return json.loads((self.path / name).read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        return f"MLWorkspace({self.path})"
