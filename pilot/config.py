"""Runtime configuration and shared paths.

No secrets live here. API keys (if you enable a real provider) are read from the
environment at call time and are NEVER persisted. There is deliberately no
credential storage anywhere in Pilot — you log into sites manually in the
persistent browser profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Project root = the directory that contains the `pilot` package.
ROOT = Path(__file__).resolve().parent.parent

PROFILES_DIR = ROOT / "profiles"
RUNS_DIR = ROOT / "runs"
RECIPES_DIR = ROOT / "recipes"
TASKS_DIR = ROOT / "tasks"

DEFAULT_PROFILE = PROFILES_DIR / "default"

# A small marker file the UI writes once the user has acknowledged the legal/
# responsible-use notice. Lives outside version control (under profiles/).
ACK_FILE = PROFILES_DIR / ".acknowledged"


class ApprovalMode(str, Enum):
    """How aggressively the loop pauses for human confirmation."""

    AUTONOMOUS = "autonomous"   # run through; stop only on errors
    CHECKPOINT = "checkpoint"   # DEFAULT: pause only on `risk` actions
    STEP = "step"               # confirm every action


@dataclass
class Settings:
    """Tunable knobs for a browser session / agent run."""

    profile_dir: Path = DEFAULT_PROFILE
    headed: bool = True                       # headed by default; tests force headless
    viewport_width: int = 1280
    viewport_height: int = 900
    approval_mode: ApprovalMode = ApprovalMode.CHECKPOINT
    max_steps: int = 40
    # Optional polite delay between actions (seconds). 0 = no extra delay.
    action_delay: float = 0.0
    # Token budget for the DOM summary handed to the model.
    dom_token_budget: int = 4000
    # Default model provider name ("stub" | "anthropic" | "openai").
    provider: str = "stub"
    # Run artifact directory (set per-run).
    run_dir: Path | None = None

    def ensure_dirs(self) -> None:
        for d in (PROFILES_DIR, RUNS_DIR, RECIPES_DIR):
            d.mkdir(parents=True, exist_ok=True)


def acknowledged() -> bool:
    """True once the user has ticked the first-run responsible-use checkbox."""
    return ACK_FILE.exists()


def set_acknowledged() -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    ACK_FILE.write_text("acknowledged\n", encoding="utf-8")


def env_api_key(provider: str) -> str | None:
    """Read a provider API key from the environment at call time only."""
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
    }.get(provider)
