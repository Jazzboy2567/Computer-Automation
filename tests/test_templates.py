"""Templates + offline demo pipeline.

Validates that every shipped task template parses into a `Task`, and that the
offline demo (fixture + StubProvider) runs the full runner -> output pipeline and
produces a report with the expected items — no network, no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FIXTURES, fixture_url
from pilot.config import TASKS_DIR
from pilot.runner import run_task
from pilot.tasks import Task

ROOT = Path(__file__).resolve().parent.parent


def test_all_templates_parse():
    files = list((ROOT / "tasks").glob("*.json"))
    assert files, "expected task templates under tasks/"
    for p in files:
        t = Task.model_validate_json(p.read_text(encoding="utf-8"))
        assert t.name and t.goal and t.sites

    # The price-compare demo is offline-runnable (stub script + recipe).
    demo = Task.load("price_compare_books.json")
    assert demo.script is not None
    assert demo.recipe
    assert demo.sort == ["price:asc"]


@pytest.mark.asyncio
async def test_offline_demo_pipeline(settings):
    """The whole runner -> output pipeline against the bundled fixture."""
    url = fixture_url("books.html")
    task = Task(
        name="Offline demo",
        goal="Collect every book's title and price, cheapest first.",
        sites=[url],
        output_schema=["title", "price", "in_stock", "url"],
        sort=["price:asc"],
        provider="stub",
        script=[
            {"type": "goto", "url": url, "reasoning": "open catalog"},
            {"type": "extract", "selector": "article.product_pod",
             "fields": {"title": "h3 a", "price": "p.price_color",
                        "url": "h3 a@href", "in_stock": "p.instock.availability"},
             "store_as": "books", "reasoning": "extract"},
            {"type": "done", "done": True, "reasoning": "done"},
        ],
    )
    result, paths = await run_task(task, settings, record_recipe=False)

    assert result.ok, result.message
    assert len(result.extracted["books"]) == 8
    md = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "## Comparison" in md
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
