"""Smoke test 1 — DOM summary quality.

Load the demo fixture, call ``get_dom_summary`` and assert it returns the
expected interactive elements (product links + add-to-cart + checkout) with
stable refs and correct visible / needs-scroll flags. A canonical snapshot of
the summary is diffed on later runs (delete it to regenerate intentionally).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FIXTURES, fixture_url

SNAPSHOT = FIXTURES / "books_summary.snapshot.txt"


def _canonical(summary) -> str:
    """ref + role + name + viewport flag, one line each (stable across runs)."""
    lines = [f"{summary.title}"]
    for e in summary.elements:
        flag = "visible" if e.in_viewport else "needs-scroll"
        lines.append(f"{e.ref} [{e.role}] {e.name!r} ({flag})")
    return "\n".join(lines) + "\n"


@pytest.mark.asyncio
async def test_dom_summary_quality(browser):
    await browser.goto(fixture_url("books.html"))
    summary = await browser.get_dom_summary(max_tokens=8000)

    names = [e.name for e in summary.elements]
    roles = [e.role for e in summary.elements]

    # The 8 product title links are present, by accessible name.
    assert "A Light in the Attic" in names
    assert "Sapiens: A Brief History of Humankind" in names
    assert roles.count("link") >= 8  # 8 titles (+ image links + next)

    # Add-to-basket buttons and the checkout button are captured as buttons.
    assert sum(1 for e in summary.elements if e.name == "Add to basket") == 8
    assert any(e.name == "Proceed to checkout" and e.role == "button" for e in summary.elements)

    # Prices are captured as readable text nodes.
    assert any(e.kind == "text" and "£51.77" in e.name for e in summary.elements)

    # data-testid drives the top-priority locator for the add buttons.
    add1 = next(e for e in summary.elements if e.attrs.get("href") is None and e.name == "Add to basket")
    assert any(loc.startswith("css=[data-testid=") for loc in add1.locators)

    # Viewport flags: the first product is visible; something is below the fold.
    first_link = next(e for e in summary.elements if e.name == "A Light in the Attic")
    assert first_link.in_viewport is True
    assert any(e.in_viewport is False for e in summary.elements), "expected needs-scroll elements"

    # Refs are stable: a second snapshot of the unchanged page yields identical refs.
    again = await browser.get_dom_summary(max_tokens=8000)
    assert [e.ref for e in summary.elements] == [e.ref for e in again.elements]
    assert [e.name for e in summary.elements] == [e.name for e in again.elements]

    # Snapshot diff (regenerates on first run if absent).
    canonical = _canonical(summary)
    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(canonical, encoding="utf-8")
        pytest.skip("wrote new DOM-summary snapshot; re-run to diff against it")
    assert canonical == SNAPSHOT.read_text(encoding="utf-8"), (
        "DOM summary changed vs snapshot. If intentional, delete "
        f"{SNAPSHOT} and re-run."
    )
