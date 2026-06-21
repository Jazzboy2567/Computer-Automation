"""Shared pytest fixtures for the Pilot smoke tests.

All tests run headless and offline against the bundled HTML fixtures (served via
``file://``), so they are deterministic and CI-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Ensure the project root is importable when running `pytest` from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pilot.browser import Browser  # noqa: E402
from pilot.config import ApprovalMode, Settings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_url(name: str) -> str:
    """file:// URL for a bundled HTML fixture."""
    return (FIXTURES / name).as_uri()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated headless settings using a temp profile + run dir."""
    s = Settings(
        profile_dir=tmp_path / "profile",
        headed=False,
        viewport_width=1280,
        viewport_height=900,
        approval_mode=ApprovalMode.CHECKPOINT,
        max_steps=20,
        provider="stub",
    )
    s.run_dir = tmp_path / "run"
    s.run_dir.mkdir(parents=True, exist_ok=True)
    return s


@pytest_asyncio.fixture
async def browser(settings: Settings):
    """A started, headless Browser bound to the isolated profile."""
    b = Browser(settings)
    await b.start()
    try:
        yield b
    finally:
        await b.close()
