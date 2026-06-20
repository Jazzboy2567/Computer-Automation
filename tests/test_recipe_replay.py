"""Smoke test 3 — recipe record & replay.

The first run records a recipe; a second run replays it deterministically with
ZERO provider calls and produces identical extracted output.
"""

from __future__ import annotations

import pytest

from conftest import fixture_url
from pilot.agent import Agent
from pilot.providers.stub import StubProvider
from pilot.recipes import RecipeStore, build_recipe
from test_e2e_stub import _script


@pytest.mark.asyncio
async def test_recipe_record_and_replay(browser, settings, tmp_path):
    url = fixture_url("books.html")
    goal = "Collect every book's title and price"

    # --- record: a normal (stub) run ---
    rec_provider = StubProvider(_script(url))
    agent = Agent(browser, rec_provider, settings)
    first = await agent.run(goal)
    assert first.ok

    store = RecipeStore(directory=tmp_path / "recipes")
    recipe = build_recipe("books-demo", goal, [url], first.steps, provider="stub")
    store.save(recipe)
    assert store.exists("books-demo")

    # --- replay: deterministic, no provider calls ---
    fallback = StubProvider([])  # would be called only if a step failed
    replay_agent = Agent(browser, fallback, settings)
    loaded = store.load("books-demo")
    second = await replay_agent.replay(loaded, fallback_provider=None)

    assert second.ok, second.message
    assert fallback.calls == [], "replay must not call any provider"
    assert second.extracted == first.extracted, "replay output must be identical"

    # The recipe captured concrete, replayable steps (goto + extract).
    types = [s.action.type.value for s in loaded.steps]
    assert "goto" in types and "extract" in types
