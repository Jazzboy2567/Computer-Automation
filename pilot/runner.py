"""High-level orchestration: run a Task end-to-end.

Ties together browser -> agent -> recipe -> output. Used by both the web server
and the CLI/tests. Recipe logic lives here: if a recipe exists for the task we
REPLAY it (no model); otherwise we run the model loop and, on success, RECORD a
recipe for next time.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .agent import Agent, ApprovalCb, EventCb
from .browser import Browser
from .config import RUNS_DIR, Settings
from .output import write_run_outputs
from .providers import get_provider
from .providers.base import Provider
from .recipes import RecipeStore, build_recipe
from .schemas import RunResult
from .tasks import Task


def new_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    d = RUNS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_provider_for(task: Task, settings: Settings) -> Provider:
    """Pick the provider. A task with a `script` runs offline via StubProvider."""
    name = task.provider or settings.provider
    if task.script is not None and name in (None, "stub"):
        return get_provider("stub", script=task.script)
    # A runtime override (settings, from CLI/UI) wins over the task's model,
    # which wins over the provider's built-in default.
    model = settings.model or task.model
    if model:
        return get_provider(name, model=model)
    return get_provider(name)


async def run_task(
    task: Task,
    settings: Settings,
    on_event: Optional[EventCb] = None,
    approval: Optional[ApprovalCb] = None,
    use_recipe: bool = True,
    record_recipe: bool = True,
) -> tuple[RunResult, dict[str, str]]:
    """Execute a task and write its report. Returns (result, artifact paths)."""
    settings.ensure_dirs()
    settings.run_dir = new_run_dir()

    browser = Browser(settings)
    await browser.start()
    try:
        provider = build_provider_for(task, settings)
        agent = Agent(browser, provider, settings, on_event=on_event, approval=approval)

        store = RecipeStore()
        recipe = store.load(task.recipe) if (task.recipe and use_recipe) else None

        if recipe is not None:
            # Replay deterministically; fall back to the model only on failure
            # (and only if the configured provider can actually call a model).
            fallback = provider if provider.name != "stub" else None
            result = await agent.replay(recipe, fallback_provider=fallback)
        else:
            result = await agent.run(task.goal, keywords=task.keywords or None)
            if result.ok and task.recipe and record_recipe:
                rec = build_recipe(task.recipe, task.goal, task.sites, result.steps, provider.name)
                store.save(rec)

        paths = write_run_outputs(result, task, settings.run_dir)
    finally:
        await browser.close()

    return result, paths
