"""Orchestrate an ML goal end-to-end: plan -> workspace -> train -> report.

This is the entry point for "the user asks for something and ML produces the
result". The LLM (if used) only runs in `planner.plan`; everything that creates
the result is the foreground engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from . import datasets, engine
from .planner import get_planner
from .spec import MLResult, MLTaskSpec
from .workspace import MLWorkspace

EventCb = Callable[[dict[str, Any]], None]


def _emit(cb: Optional[EventCb], **event: Any) -> None:
    if cb:
        cb(event)


def _build_report(
    goal: str, source: str, note: Optional[str],
    profile: dict[str, Any], spec: MLTaskSpec, result: MLResult,
) -> str:
    lines = [
        f"# ML goal: {goal}", "",
        f"**Result:** {result.task_type} — {result.headline()}  ",
        f"**Planner:** {spec.planner}{' (' + spec.notes + ')' if spec.notes else ''}  ",
        f"**Data:** {source}" + (f" — _{note}_" if note else "") + "  ",
        f"**Rows × cols:** {profile['n_rows']} × {profile['n_cols']}  ",
        f"**Model:** {result.model}",
        "",
        "## Task spec", "",
        "```json",
        spec.model_dump_json(indent=2),
        "```",
        "",
        "## Performance", "",
        "| metric | value |", "| --- | --- |",
    ]
    for k, v in result.metrics.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    if result.feature_importances:
        lines += ["## Top features", "", "| feature | importance |", "| --- | --- |"]
        for k, v in result.feature_importances.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    if result.extra.get("cluster_sizes"):
        lines += ["## Cluster sizes", "", "| cluster | n |", "| --- | --- |"]
        for k, v in result.extra["cluster_sizes"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += [
        "## Artifacts", "",
        f"- model: `{result.model_path}`",
        f"- predictions: `{result.predictions_path}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_ml_goal(
    goal: str,
    data_path: Optional[str | Path] = None,
    target: Optional[str] = None,
    planner: str = "auto",
    base_dir: Optional[Path] = None,
    on_event: Optional[EventCb] = None,
) -> tuple[MLResult, MLWorkspace]:
    """Run one ML goal in its own workspace and return (result, workspace)."""
    ws = MLWorkspace.create(goal, base_dir=base_dir)
    _emit(on_event, event="workspace", path=str(ws.path))

    df, source, note = datasets.load_dataset(data_path)
    df.to_csv(ws.data_dir / "dataset.csv", index=False)  # snapshot for reproducibility
    profile = datasets.profile_dataframe(df)
    ws.write_json("profile.json", profile)
    _emit(on_event, event="data", source=source, rows=profile["n_rows"], cols=profile["n_cols"], note=note)

    spec = get_planner(planner).plan(goal, profile, target_hint=target)
    if target and spec.task_type in ("classification", "regression"):
        spec.target = target
    ws.write_json("spec.json", spec.model_dump())
    _emit(on_event, event="plan", spec=spec.short(), planner=spec.planner)

    result = engine.run(spec, df, ws)
    ws.write_json("metrics.json", result.model_dump())
    _emit(on_event, event="result", headline=result.headline(), model=result.model)

    report = _build_report(goal, source, note, profile, spec, result)
    ws.write_text("report.md", report)
    _emit(on_event, event="report", path=str(ws.path / "report.md"))
    return result, ws
