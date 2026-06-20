"""Action loop + risk classifier + approval modes + kill/pause.

The loop is: perceive -> decide -> (maybe confirm) -> act -> record, until the
goal is met or ``max_steps`` is reached. Confirmation is gated on the action's
*risk type*, not a counter:

* ``autonomous`` — run through; stop only on errors.
* ``checkpoint`` (DEFAULT) — pause ONLY on ``risk`` actions.
* ``step`` — confirm every action.

A persistent kill/pause control stops the loop immediately.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .browser import Browser
from .config import ApprovalMode, Settings
from .perception import perceive
from .providers.base import Provider
from .recipes import Recipe
from .schemas import (
    Action,
    ActionType,
    PageState,
    RiskLevel,
    RunResult,
    StepRecord,
)

log = logging.getLogger("pilot.agent")

# Words in a click target's accessible name that mark it as a `risk` action:
# spends money, submits/sends/posts, deletes, or changes account settings.
RISK_KEYWORDS = (
    "checkout", "check out", "place order", "buy now", "buy", "purchase", "pay",
    "payment", "complete order", "order now", "submit", "confirm", "delete",
    "remove", "unsubscribe", "subscribe", "send", "post", "publish", "save",
    "apply", "book now", "reserve", "donate", "transfer", "withdraw", "bid",
    "sign out", "log out", "logout", "change password", "update settings",
)

# read-only action types: these never spend money or mutate account state.
_READ_TYPES = {
    ActionType.GOTO,
    ActionType.SCROLL,
    ActionType.EXTRACT,
    ActionType.WAIT,
    ActionType.SCREENSHOT,
    ActionType.DONE,
    ActionType.TYPE,  # typing into a field is itself reversible/non-committing
}


def classify_risk(
    action: Action,
    page_state: Optional[PageState] = None,
    target_name: Optional[str] = None,
) -> RiskLevel:
    """Tag an action ``read`` or ``risk``.

    Everything except a click is read-only. A click is ``risk`` when its target's
    accessible name contains a money/submit/delete/settings keyword — e.g.
    "Proceed to checkout", "Place order", "Delete account".
    """
    if action.type in _READ_TYPES:
        return RiskLevel.READ
    if action.type is ActionType.CLICK:
        name = target_name or ""
        if not name and page_state and action.ref:
            el = page_state.dom_summary.by_ref(action.ref)
            if el:
                name = el.name
        low = name.lower()
        if any(kw in low for kw in RISK_KEYWORDS):
            return RiskLevel.RISK
    return RiskLevel.READ


def needs_confirmation(mode: ApprovalMode, risk: RiskLevel) -> bool:
    """Whether the loop must pause for human confirmation before acting."""
    if mode is ApprovalMode.AUTONOMOUS:
        return False
    if mode is ApprovalMode.STEP:
        return True
    # checkpoint: only risk actions.
    return risk is RiskLevel.RISK


# Callbacks the host (server / CLI / tests) can plug in.
EventCb = Callable[[dict[str, Any]], Awaitable[None] | None]
ApprovalCb = Callable[[Action, RiskLevel], Awaitable[bool]]


async def _auto_approve(action: Action, risk: RiskLevel) -> bool:
    return True


class Agent:
    """Drives a browser toward a goal using a provider (or a recorded recipe)."""

    def __init__(
        self,
        browser: Browser,
        provider: Provider,
        settings: Settings,
        on_event: Optional[EventCb] = None,
        approval: Optional[ApprovalCb] = None,
    ):
        self.browser = browser
        self.provider = provider
        self.settings = settings
        self.on_event = on_event
        self.approval = approval or _auto_approve

        # Loop control. _run_flag set == running; clear == paused.
        self._run_flag = asyncio.Event()
        self._run_flag.set()
        self._stop = asyncio.Event()

    # ------------------------------------------------------------- controls
    def pause(self) -> None:
        self._run_flag.clear()

    def resume(self) -> None:
        self._run_flag.set()

    def stop(self) -> None:
        """Kill the loop ASAP (also unblocks a paused loop)."""
        self._stop.set()
        self._run_flag.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    async def _gate(self) -> None:
        """Block here while paused (unless stopping)."""
        if not self._stop.is_set():
            await self._run_flag.wait()

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        res = self.on_event(event)
        if asyncio.iscoroutine(res):
            await res

    # ------------------------------------------------------------- execution
    async def _execute(
        self,
        action: Action,
        index: int,
        risk: RiskLevel,
        extracted: dict[str, Any],
        page_state: Optional[PageState] = None,
        run_dir: Optional[Path] = None,
    ) -> StepRecord:
        """Dispatch a single action to the browser and record the outcome."""
        assert self.browser.page
        rec = StepRecord(
            index=index,
            action=action,
            risk=risk,
            url_before=self.browser.page.url,
            perception_mode=page_state.perception_mode if page_state else "dom",
        )
        try:
            if self.settings.action_delay:
                await asyncio.sleep(self.settings.action_delay)

            t = action.type
            if t is ActionType.GOTO:
                await self.browser.goto(action.url or "about:blank")
            elif t is ActionType.CLICK:
                if action.ref:
                    rec.used_locator = await self.browser.click(action.ref)
                elif action.x is not None and action.y is not None:
                    await self.browser.click_xy(action.x, action.y)  # vision fallback
                else:
                    raise ValueError("click action needs a ref or x/y")
            elif t is ActionType.TYPE:
                if not action.ref:
                    raise ValueError("type action needs a ref")
                rec.used_locator = await self.browser.type(action.ref, action.text or "")
            elif t is ActionType.SCROLL:
                await self.browser.scroll(action.direction or "down", action.amount, action.ref)
            elif t is ActionType.EXTRACT:
                data = await self.browser.extract(action.selector or "", action.fields)
                key = action.store_as or "data"
                extracted[key] = data
                rec.extracted = data
            elif t is ActionType.WAIT:
                await self.browser.wait(action.seconds, action.wait_for)
            elif t in (ActionType.DONE, ActionType.SCREENSHOT):
                pass
            rec.ok = True
        except Exception as e:  # one bad step shouldn't crash the whole run
            rec.ok = False
            rec.error = f"{type(e).__name__}: {e}"
            log.warning("step %d failed: %s", index, rec.error)
        rec.url_after = self.browser.page.url
        return rec

    async def _confirm(self, action: Action, risk: RiskLevel) -> bool:
        await self._emit(
            {"event": "awaiting_approval", "action": action.model_dump(), "risk": risk.value,
             "summary": action.short()}
        )
        approved = await self.approval(action, risk)
        await self._emit({"event": "approval_result", "approved": approved})
        return approved

    # --------------------------------------------------------------- run loop
    async def run(self, goal: str, keywords: Optional[list[str]] = None) -> RunResult:
        """Model-driven loop. Returns a RunResult with full history + extraction."""
        run_dir = self.settings.run_dir
        history: list[StepRecord] = []
        extracted: dict[str, Any] = {}
        ok = True
        message = None

        for step in range(self.settings.max_steps):
            if self._stop.is_set():
                message = "stopped by user"
                ok = False
                break
            await self._gate()
            if self._stop.is_set():
                message = "stopped by user"
                ok = False
                break

            page_state = await perceive(self.browser, run_dir, step, keywords)
            await self._emit(
                {"event": "perception", "step": step, "url": page_state.url,
                 "mode": page_state.perception_mode, "elements": len(page_state.dom_summary.elements),
                 "screenshot": page_state.screenshot_path}
            )

            action = await self.provider.decide(goal, page_state, history)
            risk = classify_risk(action, page_state)
            await self._emit(
                {"event": "decision", "step": step, "action": action.model_dump(),
                 "summary": action.short(), "risk": risk.value, "reasoning": action.reasoning}
            )

            if needs_confirmation(self.settings.approval_mode, risk):
                if not await self._confirm(action, risk):
                    # Declining a gated action aborts the run (never proceed past a gate).
                    history.append(
                        StepRecord(index=step, action=action, risk=risk, ok=False,
                                   error="declined by user")
                    )
                    ok = False
                    message = "stopped: action declined at approval gate"
                    break

            rec = await self._execute(action, step, risk, extracted, page_state, run_dir)
            rec.screenshot_path = page_state.screenshot_path
            history.append(rec)
            await self._emit({"event": "executed", "step": step, "ok": rec.ok, "error": rec.error})

            if not rec.ok:
                ok = False
                message = f"stopped on error at step {step}: {rec.error}"
                break
            if action.done or action.type is ActionType.DONE:
                message = "goal complete"
                break
        else:
            message = "reached max steps"
            ok = False

        result = RunResult(
            goal=goal, ok=ok, steps=history, extracted=extracted,
            run_dir=str(run_dir) if run_dir else None, message=message,
        )
        await self._emit({"event": "finished", "ok": ok, "message": message})
        return result

    # --------------------------------------------------------------- replay
    async def replay(
        self,
        recipe: Recipe,
        fallback_provider: Optional[Provider] = None,
    ) -> RunResult:
        """Replay a recorded recipe deterministically — NO model calls.

        If a step fails (locator missing / layout changed) and a
        ``fallback_provider`` is supplied, the agent re-perceives the live page
        and asks the model for the next action, then continues. With no fallback,
        a failed step stops the run (so the test can assert zero provider calls).
        """
        run_dir = self.settings.run_dir
        history: list[StepRecord] = []
        extracted: dict[str, Any] = {}
        ok = True
        message = "replayed from recipe"

        for i, step in enumerate(recipe.steps):
            if self._stop.is_set():
                ok = False
                message = "stopped by user"
                break
            await self._gate()

            action = step.action
            risk = step.risk
            await self._emit(
                {"event": "replay_step", "step": i, "summary": action.short(), "risk": risk.value}
            )

            if needs_confirmation(self.settings.approval_mode, risk):
                if not await self._confirm(action, risk):
                    history.append(
                        StepRecord(index=i, action=action, risk=risk, ok=False,
                                   error="declined by user")
                    )
                    ok = False
                    message = "stopped: action declined at approval gate"
                    break

            rec = await self._replay_one(action, step.locator, i, risk, extracted)
            history.append(rec)
            await self._emit({"event": "executed", "step": i, "ok": rec.ok, "error": rec.error})

            if not rec.ok:
                if fallback_provider is None:
                    ok = False
                    message = f"replay failed at step {i}: {rec.error} (no fallback)"
                    break
                # Re-plan from the live page, update history, keep going.
                log.info("replay step %d failed; falling back to model re-plan", i)
                await self._emit({"event": "replay_fallback", "step": i})
                page_state = await perceive(self.browser, run_dir, i)
                action = await fallback_provider.decide(recipe.goal, page_state, history)
                risk = classify_risk(action, page_state)
                rec2 = await self._execute(action, i, risk, extracted, page_state, run_dir)
                history.append(rec2)
                if not rec2.ok:
                    ok = False
                    message = f"replay + fallback both failed at step {i}"
                    break

        result = RunResult(
            goal=recipe.goal, ok=ok, steps=history, extracted=extracted,
            run_dir=str(run_dir) if run_dir else None, message=message,
        )
        await self._emit({"event": "finished", "ok": ok, "message": message})
        return result

    async def _replay_one(
        self,
        action: Action,
        locator: Optional[str],
        index: int,
        risk: RiskLevel,
        extracted: dict[str, Any],
    ) -> StepRecord:
        """Execute one recorded step using its durable locator (no DOM snapshot)."""
        assert self.browser.page
        rec = StepRecord(index=index, action=action, risk=risk,
                         url_before=self.browser.page.url, used_locator=locator)
        try:
            if self.settings.action_delay:
                await asyncio.sleep(self.settings.action_delay)
            t = action.type
            if t is ActionType.GOTO:
                await self.browser.goto(action.url or "about:blank")
            elif t is ActionType.CLICK:
                if not locator:
                    raise ValueError("recipe click step has no locator")
                await self.browser.click_locator(locator)
            elif t is ActionType.TYPE:
                if not locator:
                    raise ValueError("recipe type step has no locator")
                await self.browser.type_locator(locator, action.text or "")
            elif t is ActionType.SCROLL:
                await self.browser.scroll(action.direction or "down", action.amount)
            elif t is ActionType.EXTRACT:
                data = await self.browser.extract(action.selector or "", action.fields)
                extracted[action.store_as or "data"] = data
                rec.extracted = data
            elif t is ActionType.WAIT:
                await self.browser.wait(action.seconds, action.wait_for)
            elif t in (ActionType.DONE, ActionType.SCREENSHOT):
                pass
            rec.ok = True
        except Exception as e:
            rec.ok = False
            rec.error = f"{type(e).__name__}: {e}"
        rec.url_after = self.browser.page.url
        return rec
