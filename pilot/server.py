"""FastAPI server + small web UI.

The UI is how you drive Pilot and watch it work: pick a task, choose an approval
mode, start/pause/kill the run, approve gated (risk) actions, and stream live
events + screenshots. A first-run responsible-use notice must be acknowledged
before any run can start.

Run with:  uvicorn pilot.server:app --reload   (or: python -m pilot.server)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ApprovalMode, RUNS_DIR, Settings, acknowledged, set_acknowledged
from .recipes import RecipeStore
from .schemas import Action, RiskLevel
from .tasks import Task, TASKS_DIR

log = logging.getLogger("pilot.server")
logging.basicConfig(level=logging.INFO)

WEB_DIR = Path(__file__).with_name("web")

app = FastAPI(title="Pilot", version="0.1.0")


# ---------------------------------------------------------------------------
# Run session: a single active run with an event queue + approval future.
# ---------------------------------------------------------------------------


class RunSession:
    """Holds the state of the one active run so the UI can observe/control it."""

    def __init__(self):
        self.agent = None  # type: ignore[assignment]
        self.task: Optional[asyncio.Task] = None
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_approval: Optional[asyncio.Future[bool]] = None
        self.active = False

    async def emit(self, event: dict[str, Any]) -> None:
        await self.queue.put(event)

    async def approval(self, action: Action, risk: RiskLevel) -> bool:
        """Block the agent until the UI answers the approval prompt."""
        loop = asyncio.get_event_loop()
        self._pending_approval = loop.create_future()
        await self.emit(
            {"event": "awaiting_approval", "summary": action.short(), "risk": risk.value}
        )
        return await self._pending_approval

    def resolve_approval(self, approved: bool) -> bool:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result(approved)
            self._pending_approval = None
            return True
        return False


SESSION = RunSession()


# ---------------------------------------------------------------------------
# Models for request bodies
# ---------------------------------------------------------------------------


class StartBody(BaseModel):
    task_file: Optional[str] = None     # a tasks/*.json filename
    goal: Optional[str] = None          # ad-hoc goal (ignored if task_file set)
    start_url: Optional[str] = None     # optional first navigation for ad-hoc goals
    approval_mode: str = "checkpoint"
    provider: str = "stub"
    headed: bool = True
    action_delay: float = 0.0


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# First-run acknowledgement (legal / responsible-use gate)
# ---------------------------------------------------------------------------


@app.get("/api/ack")
async def get_ack() -> dict[str, bool]:
    return {"acknowledged": acknowledged()}


@app.post("/api/ack")
async def post_ack(body: dict[str, bool]) -> dict[str, bool]:
    if body.get("accept"):
        set_acknowledged()
    return {"acknowledged": acknowledged()}


# ---------------------------------------------------------------------------
# Tasks & recipes listing
# ---------------------------------------------------------------------------


@app.get("/api/tasks")
async def list_tasks() -> dict[str, list[dict[str, Any]]]:
    out = []
    if TASKS_DIR.exists():
        for p in sorted(TASKS_DIR.glob("*.json")):
            try:
                t = Task.model_validate_json(p.read_text(encoding="utf-8"))
                out.append({"file": p.name, "name": t.name, "goal": t.goal,
                            "sites": t.sites, "provider": t.provider,
                            "has_script": t.script is not None})
            except Exception as e:  # pragma: no cover
                out.append({"file": p.name, "name": p.name, "error": str(e)})
    return {"tasks": out}


@app.get("/api/recipes")
async def list_recipes() -> dict[str, list[str]]:
    return {"recipes": RecipeStore().list_names()}


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@app.post("/api/start")
async def start_run(body: StartBody) -> JSONResponse:
    if not acknowledged():
        raise HTTPException(status_code=403, detail="Acknowledge the responsible-use notice first.")
    if SESSION.active:
        raise HTTPException(status_code=409, detail="A run is already in progress.")

    # Build the Task (from file or an ad-hoc goal).
    if body.task_file:
        task = Task.load(body.task_file)
    elif body.goal:
        script = None
        if body.start_url:
            # Give ad-hoc goals a first navigation so the model starts somewhere.
            script = None
        task = Task(name="Ad-hoc", goal=body.goal,
                    sites=[body.start_url] if body.start_url else [])
    else:
        raise HTTPException(status_code=400, detail="Provide a task_file or a goal.")

    try:
        mode = ApprovalMode(body.approval_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Bad approval_mode {body.approval_mode!r}")

    settings = Settings(
        headed=body.headed,
        approval_mode=mode,
        provider=body.provider,
        action_delay=body.action_delay,
    )

    # Reset session queue for a fresh run.
    SESSION.queue = asyncio.Queue()
    SESSION.active = True
    SESSION.task = asyncio.create_task(_run(task, settings, body))
    return JSONResponse({"started": True, "task": task.name})


async def _run(task: Task, settings: Settings, body: StartBody) -> None:
    """Background coroutine that actually drives the run (imported lazily)."""
    from .agent import Agent
    from .browser import Browser
    from .output import write_run_outputs
    from .providers import get_provider
    from .recipes import RecipeStore, build_recipe
    from .runner import new_run_dir

    settings.ensure_dirs()
    settings.run_dir = new_run_dir()
    browser = Browser(settings)
    try:
        await browser.start()
        # Ad-hoc goal with a start URL: navigate there before the loop.
        if body.goal and body.start_url:
            await browser.goto(body.start_url)

        if task.script is not None and (task.provider or settings.provider) in (None, "stub"):
            provider = get_provider("stub", script=task.script)
        else:
            provider = get_provider(task.provider or settings.provider)

        agent = Agent(browser, provider, settings,
                      on_event=SESSION.emit, approval=SESSION.approval)
        SESSION.agent = agent

        store = RecipeStore()
        recipe = store.load(task.recipe) if task.recipe else None
        if recipe is not None:
            fallback = provider if provider.name != "stub" else None
            result = await agent.replay(recipe, fallback_provider=fallback)
        else:
            result = await agent.run(task.goal, keywords=task.keywords or None)
            if result.ok and task.recipe:
                store.save(build_recipe(task.recipe, task.goal, task.sites,
                                        result.steps, provider.name))

        paths = write_run_outputs(result, task, settings.run_dir)
        await SESSION.emit({"event": "report", "ok": result.ok, "message": result.message,
                            "paths": paths, "run_dir": str(settings.run_dir)})
    except Exception as e:  # pragma: no cover - surfaced to the UI
        log.exception("run failed")
        await SESSION.emit({"event": "error", "fatal": True, "error": f"{type(e).__name__}: {e}"})
    finally:
        await browser.close()
        SESSION.active = False
        await SESSION.emit({"event": "closed"})


@app.post("/api/approve")
async def approve(body: dict[str, bool]) -> dict[str, bool]:
    ok = SESSION.resolve_approval(bool(body.get("approved")))
    return {"resolved": ok}


@app.post("/api/pause")
async def pause() -> dict[str, str]:
    if SESSION.agent:
        SESSION.agent.pause()
    return {"state": "paused"}


@app.post("/api/resume")
async def resume() -> dict[str, str]:
    if SESSION.agent:
        SESSION.agent.resume()
    return {"state": "running"}


@app.post("/api/stop")
async def stop() -> dict[str, str]:
    """The persistent kill button: stop the loop immediately."""
    if SESSION.agent:
        SESSION.agent.stop()
    # Unblock any pending approval so the loop can exit.
    SESSION.resolve_approval(False)
    return {"state": "stopping"}


@app.get("/api/events")
async def events() -> StreamingResponse:
    """Server-Sent Events stream of run events for the UI."""

    async def gen():
        while True:
            event = await SESSION.queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("event") == "closed":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/screenshot")
async def screenshot(path: str) -> Any:
    """Serve a run screenshot by path (must be inside runs/)."""
    from fastapi.responses import FileResponse

    p = Path(path).resolve()
    if RUNS_DIR.resolve() not in p.parents:
        raise HTTPException(status_code=403, detail="Path outside runs/.")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(str(p))


def main() -> None:
    import uvicorn

    uvicorn.run("pilot.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
