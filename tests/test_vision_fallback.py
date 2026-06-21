"""Smoke test 5 — vision fallback trigger.

On a canvas page the DOM summary is empty, so perception must switch to VISION
mode and log it. On the normal DOM page it stays in DOM mode.
"""

from __future__ import annotations

import logging

import pytest

from conftest import fixture_url
from pilot.perception import perceive


@pytest.mark.asyncio
async def test_vision_fallback_on_canvas(browser, settings, caplog):
    await browser.goto(fixture_url("canvas.html"))

    with caplog.at_level(logging.WARNING, logger="pilot.perception"):
        state = await perceive(browser, run_dir=settings.run_dir, step_index=0)

    assert state.perception_mode == "vision"
    assert state.perception_note and "vision fallback" in state.perception_note
    assert "canvas" in state.perception_note.lower()
    # It was logged.
    assert any("VISION" in r.message for r in caplog.records)
    # A screenshot was still captured (vision needs the image).
    assert state.screenshot_path


@pytest.mark.asyncio
async def test_dom_mode_on_normal_page(browser, settings):
    await browser.goto(fixture_url("books.html"))
    state = await perceive(browser, run_dir=settings.run_dir, step_index=0)
    assert state.perception_mode == "dom"
    assert state.perception_note is None
    assert len(state.dom_summary.elements) > 0
