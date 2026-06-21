"""Tasks & comparison.

A *task* = goal + target sites + output schema (+ optional sort + optional
scripted steps for deterministic/stub runs). Comparison and ranking are done in
PLAIN CODE here — no model — by normalizing extracted items and sorting them
(price low->high, in-stock first, job keyword-match, ...). The model is only
used for the perception/decision loop, never for the deterministic ranking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import TASKS_DIR


class Task(BaseModel):
    """A reusable task definition (loaded from ``tasks/*.json``)."""

    name: str
    goal: str
    sites: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # Names of the fields each extracted item is expected to have.
    output_schema: list[str] = Field(default_factory=list)
    # Sort spec, e.g. ["price:asc", "in_stock:desc"] or ["match:desc"].
    sort: list[str] = Field(default_factory=list)
    provider: Optional[str] = None        # override default provider
    model: Optional[str] = None           # override model id, e.g. "claude-sonnet-4-6"
    recipe: Optional[str] = None          # recipe name to record/replay
    # Optional scripted actions => the task can run fully offline via StubProvider.
    script: Optional[list[dict[str, Any]]] = None

    @staticmethod
    def load(path: str | Path) -> "Task":
        p = Path(path)
        if not p.exists() and not p.is_absolute():
            p = TASKS_DIR / p
        return Task.model_validate_json(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Plain-code normalization, comparison and ranking
# ---------------------------------------------------------------------------

_PRICE_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_price(value: Any) -> Optional[float]:
    """Pull a number out of a price string like ``"£51.77"`` or ``"$1,299.00"``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _PRICE_RE.search(str(value).replace(",", ""))
    return float(m.group()) if m else None


def parse_bool(value: Any) -> bool:
    """Interpret stock/availability-ish text as a boolean."""
    if isinstance(value, bool):
        return value
    s = str(value).lower()
    if any(w in s for w in ("in stock", "available", "yes", "true")):
        return True
    if any(w in s for w in ("out of stock", "unavailable", "sold out", "no", "false")):
        return False
    return bool(value)


def match_score(item: dict[str, Any], keywords: list[str]) -> int:
    """Count keyword hits across an item's text fields (for job ranking)."""
    if not keywords:
        return 0
    hay = " ".join(str(v) for v in item.values()).lower()
    return sum(1 for k in keywords if k.lower() in hay)


def normalize_items(
    items: list[dict[str, Any]], keywords: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Add derived comparison fields (``_price``, ``_in_stock``, ``_match``)."""
    out = []
    for it in items:
        it = dict(it)
        for key in ("price", "Price", "cost"):
            if key in it:
                it["_price"] = parse_price(it[key])
                break
        for key in ("in_stock", "stock", "availability", "Availability"):
            if key in it:
                it["_in_stock"] = parse_bool(it[key])
                break
        if keywords:
            it["_match"] = match_score(it, keywords)
        out.append(it)
    return out


def rank_items(
    items: list[dict[str, Any]],
    sort: list[str],
    keywords: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Deterministically sort items by a multi-key spec like ``["price:asc"]``.

    Recognized fields map onto the derived ones: ``price`` -> ``_price``,
    ``in_stock`` -> ``_in_stock``, ``match`` -> ``_match``. Unknown fields sort
    on their raw value. Missing values sort last.
    """
    items = normalize_items(items, keywords)
    if not sort:
        return items

    field_alias = {"price": "_price", "in_stock": "_in_stock", "match": "_match"}

    # Apply keys from least to most significant (stable sort => correct order).
    for spec in reversed(sort):
        field, _, direction = spec.partition(":")
        field = field_alias.get(field, field)
        reverse = direction.lower() == "desc"

        def key(it: dict[str, Any], f=field):
            v = it.get(f)
            # Push missing values to the end regardless of direction.
            missing = v is None
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, str):
                v = v.lower()
            return (missing, v if v is not None else 0)

        items.sort(key=key, reverse=reverse)
        # Re-stabilize "missing last": pull missing to the end after a reverse sort.
        if reverse:
            present = [i for i in items if i.get(field) is not None]
            absent = [i for i in items if i.get(field) is None]
            items = present + absent
    return items


def collect_items(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a run's extracted data into a single list of item dicts.

    Each list-of-dict extraction becomes items tagged with their ``source`` key;
    list-of-string extractions are wrapped as ``{"text": ..., "source": key}``.
    """
    items: list[dict[str, Any]] = []
    for source, value in extracted.items():
        if isinstance(value, list):
            for v in value:
                if isinstance(v, dict):
                    row = dict(v)
                    row.setdefault("source", source)
                    items.append(row)
                else:
                    items.append({"text": v, "source": source})
        elif isinstance(value, dict):
            row = dict(value)
            row.setdefault("source", source)
            items.append(row)
    return items
