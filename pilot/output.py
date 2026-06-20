"""Output — markdown report (+ cross-site comparison table), JSON and CSV.

Default deliverable is a clean markdown list with links plus a comparison table.
Everything is also exported to JSON and CSV. Per-step screenshots are written by
the perception layer; here we persist the report + the structured data under
``runs/<timestamp>/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .schemas import RunResult
from .tasks import Task, collect_items, rank_items

# Column preference for the comparison table (only those present are shown).
_PREFERRED_COLS = [
    "source", "site", "title", "name", "company", "location",
    "price", "in_stock", "availability", "rating", "url", "link",
]
_LABEL_KEYS = ("title", "name", "text", "label")
_URL_KEYS = ("url", "link", "href")


def _label(item: dict[str, Any]) -> str:
    for k in _LABEL_KEYS:
        if item.get(k):
            return str(item[k])
    return next((str(v) for v in item.values() if v), "item")


def _url(item: dict[str, Any]) -> Optional[str]:
    for k in _URL_KEYS:
        if item.get(k):
            return str(item[k])
    return None


def _visible_cols(items: list[dict[str, Any]]) -> list[str]:
    present = set()
    for it in items:
        present.update(k for k in it if not k.startswith("_"))
    cols = [c for c in _PREFERRED_COLS if c in present]
    # Append any leftover non-derived columns we didn't anticipate.
    cols += sorted(c for c in present if c not in cols and not c.startswith("_"))
    return cols


def build_markdown(result: RunResult, task: Optional[Task], items: list[dict[str, Any]]) -> str:
    lines = [f"# {task.name if task else 'Pilot run'}", "", f"**Goal:** {result.goal}", ""]
    lines.append(f"**Status:** {'✅ ' + (result.message or 'ok') if result.ok else '⚠️ ' + (result.message or 'failed')}")
    lines.append(f"**Steps:** {len(result.steps)}  ")
    lines.append("")

    if not items:
        lines.append("_No items were extracted._")
        return "\n".join(lines) + "\n"

    # Comparison table.
    cols = _visible_cols(items)
    lines.append("## Comparison")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for it in items:
        row = []
        for c in cols:
            v = it.get(c, "")
            row.append("" if v is None else str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Linked bullet list.
    lines.append("## Items")
    lines.append("")
    for it in items:
        label = _label(it)
        url = _url(it)
        price = it.get("price") or it.get("_price")
        suffix = f" — {price}" if price else ""
        if url:
            lines.append(f"- [{label}]({url}){suffix}")
        else:
            lines.append(f"- {label}{suffix}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, items: list[dict[str, Any]]) -> None:
    cols = _visible_cols(items)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({c: ("" if it.get(c) is None else it.get(c)) for c in cols})


def write_run_outputs(
    result: RunResult,
    task: Optional[Task] = None,
    run_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Rank the extracted items and write report.md / report.json / report.csv.

    Returns a map of artifact name -> path. Screenshots are already saved per
    step by the perception layer (``runs/<ts>/step_NN.png``).
    """
    run_dir = Path(run_dir or result.run_dir or ".")
    run_dir.mkdir(parents=True, exist_ok=True)

    items = collect_items(result.extracted)
    sort = task.sort if task else []
    keywords = task.keywords if task else []
    items = rank_items(items, sort, keywords)

    # Drop derived (_*) fields from the persisted item rows for cleanliness.
    clean = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]

    md = build_markdown(result, task, clean)
    md_path = run_dir / "report.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = run_dir / "report.json"
    json_path.write_text(
        json.dumps(
            {
                "goal": result.goal,
                "ok": result.ok,
                "message": result.message,
                "items": clean,
                "raw_extracted": result.extracted,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path = run_dir / "report.csv"
    if clean:
        write_csv(csv_path, clean)

    paths = {"markdown": str(md_path), "json": str(json_path)}
    if clean:
        paths["csv"] = str(csv_path)
    return paths
