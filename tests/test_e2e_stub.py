"""Smoke test 2 — end-to-end loop with StubProvider (no API calls).

Scripted actions (navigate -> extract product names+prices -> done) run through
the full perceive -> decide -> act loop with the StubProvider, producing a
markdown list + JSON + CSV under the run directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from conftest import fixture_url
from pilot.agent import Agent
from pilot.output import write_run_outputs
from pilot.providers.stub import StubProvider
from pilot.tasks import Task


def _script(url: str) -> list[dict]:
    return [
        {"type": "goto", "url": url, "reasoning": "open the catalog"},
        {
            "type": "extract",
            "selector": "article.product_pod",
            "fields": {
                "title": "h3 a",
                "price": "p.price_color",
                "url": "h3 a@href",
                "in_stock": "p.instock.availability",
            },
            "store_as": "books",
            "reasoning": "extract product names and prices",
        },
        {"type": "done", "done": True, "reasoning": "collected all products"},
    ]


@pytest.mark.asyncio
async def test_e2e_stub_produces_reports(browser, settings):
    url = fixture_url("books.html")
    provider = StubProvider(_script(url))
    agent = Agent(browser, provider, settings)

    result = await agent.run("Collect every book's title and price")

    assert result.ok, result.message
    assert "books" in result.extracted
    books = result.extracted["books"]
    assert len(books) == 8
    assert {"title", "price", "url", "in_stock"} <= set(books[0])
    assert books[0]["title"] == "A Light in the Attic"

    # Write the reports (sorted cheapest-first, in code, no model).
    task = Task(
        name="Books price compare (demo)",
        goal=result.goal,
        sites=[url],
        output_schema=["title", "price", "in_stock", "url"],
        sort=["price:asc"],
    )
    paths = write_run_outputs(result, task, settings.run_dir)

    md = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "A Light in the Attic" in md
    assert "£17.93" in md  # cheapest book present
    assert "## Comparison" in md

    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert len(data["items"]) == 8
    # Plain-code ranking: cheapest first.
    assert data["items"][0]["title"] == "The Coming Woman: A Novel"

    with Path(paths["csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert "price" in rows[0]

    # Per-step screenshots were captured.
    assert list(Path(settings.run_dir).glob("step_*.png"))
