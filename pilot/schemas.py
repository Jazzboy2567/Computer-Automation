"""Shared, centralized schemas for actions, perception and results.

Everything that crosses a module boundary is a Pydantic model defined here so
that providers, the agent loop, recipes and the UI all speak the same language.
The *action schema* in particular is centralized: every provider emits the same
`Action` JSON, and prompts differ only in how they format/explain it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """The complete vocabulary of things the agent can ask the browser to do.

    Keep this list small and explicit. The risk classifier (see
    ``pilot.agent.classify_risk``) maps each type to ``read`` or ``risk``.
    """

    GOTO = "goto"          # navigate to a URL
    CLICK = "click"        # click an element by ref
    TYPE = "type"          # type text into an element by ref
    SCROLL = "scroll"      # scroll the page (up/down/to element)
    EXTRACT = "extract"    # pull structured text/data out of the page
    WAIT = "wait"          # wait for a condition / time
    SCREENSHOT = "screenshot"  # capture a screenshot (perception only)
    DONE = "done"          # goal achieved; stop the loop


class Action(BaseModel):
    """A single structured step the model (or a recipe) wants to perform.

    Fields are intentionally permissive: each `ActionType` uses a subset.
    `ref` refers to a short element id from the DOM summary (e.g. ``"e12"``).
    """

    type: ActionType
    # Element target (for click/type/scroll-to-element/extract).
    ref: Optional[str] = None
    # Free text payload (type=TYPE), or a CSS selector (type=EXTRACT),
    # or a URL (type=GOTO).
    text: Optional[str] = None
    url: Optional[str] = None
    selector: Optional[str] = None
    # Scroll controls.
    direction: Optional[Literal["up", "down", "top", "bottom", "to_element"]] = None
    amount: Optional[int] = None  # pixels, for scroll
    # Wait controls.
    seconds: Optional[float] = None
    wait_for: Optional[str] = None  # selector / "networkidle" / "load"
    # Extraction: a name to store the result under and an optional field map.
    store_as: Optional[str] = None
    fields: Optional[dict[str, str]] = None  # {field_name: css_selector_relative}
    # Whether the model thinks the goal is complete (mirrors DONE but lets any
    # action carry a done flag, matching the "next action + done flag" contract).
    done: bool = False
    # The model's short reasoning for this step (for transparency + recipes).
    reasoning: Optional[str] = None
    # Vision fallback: when targeting by pixel coordinate instead of a ref.
    x: Optional[int] = None
    y: Optional[int] = None

    def short(self) -> str:
        """One-line human description used in the UI/history/logs."""
        t = self.type.value
        if self.type is ActionType.GOTO:
            return f"goto {self.url}"
        if self.type is ActionType.CLICK:
            return f"click {self.ref or f'({self.x},{self.y})'}"
        if self.type is ActionType.TYPE:
            preview = (self.text or "")[:40]
            return f"type {self.ref!r} <- {preview!r}"
        if self.type is ActionType.SCROLL:
            return f"scroll {self.direction or 'down'}"
        if self.type is ActionType.EXTRACT:
            return f"extract -> {self.store_as or 'data'}"
        if self.type is ActionType.WAIT:
            return f"wait {self.seconds or self.wait_for}"
        if self.type is ActionType.DONE:
            return "done"
        return t


# ---------------------------------------------------------------------------
# DOM summary / perception
# ---------------------------------------------------------------------------


class DomElement(BaseModel):
    """One kept element from the DOM summary.

    `locators` is the ordered fallback chain produced by `get_dom_summary` so a
    ref can be re-found after DOM churn (data-testid/id -> role+name -> text ->
    css/xpath path). `ref` is stable within a page state for the run.
    """

    ref: str
    role: str                      # button / link / textbox / heading / text / ...
    name: str = ""                 # accessible name / visible text
    tag: str = ""                  # html tag, lowercased
    kind: Literal["interactive", "text"] = "interactive"
    attrs: dict[str, str] = Field(default_factory=dict)  # href, value, placeholder, type...
    in_viewport: bool = True
    locators: list[str] = Field(default_factory=list)    # ordered fallback chain
    frame_path: list[int] = Field(default_factory=list)  # iframe indices to reach the element
    parent_hint: Optional[str] = None

    def to_line(self) -> str:
        """Render the compact one-line form the model sees."""
        flag = "visible" if self.in_viewport else "needs-scroll"
        bits = [f"{self.ref} [{self.role}]"]
        if self.name:
            bits.append(f'"{self.name}"')
        # Surface the few attributes that help the model decide.
        for key in ("href", "value", "placeholder", "type"):
            if key in self.attrs and self.attrs[key]:
                bits.append(f"{key}={self.attrs[key]}")
        if self.parent_hint:
            bits.append(f"in:{self.parent_hint}")
        bits.append(f"({flag})")
        return "  ".join(bits)


class DomSummary(BaseModel):
    """The compact, token-budgeted view of a page handed to the model."""

    url: str
    title: str
    elements: list[DomElement] = Field(default_factory=list)
    truncated: bool = False
    total_found: int = 0           # before truncation
    approx_tokens: int = 0
    notes: list[str] = Field(default_factory=list)  # e.g. cross-origin iframe skipped

    def render(self) -> str:
        """The text block the model actually reads."""
        head = f"PAGE: {self.title}\nURL: {self.url}\n"
        lines = "\n".join(e.to_line() for e in self.elements)
        tail = ""
        if self.truncated:
            tail = (
                f"\n... [{self.total_found - len(self.elements)} more elements hidden by "
                f"token budget; scroll or request a region to see more]"
            )
        if self.notes:
            tail += "\nNOTES: " + "; ".join(self.notes)
        return f"{head}{lines}{tail}"

    def by_ref(self, ref: str) -> Optional[DomElement]:
        for e in self.elements:
            if e.ref == ref:
                return e
        return None


class ViewportInfo(BaseModel):
    width: int
    height: int
    scroll_x: int = 0
    scroll_y: int = 0
    page_height: int = 0


class PageState(BaseModel):
    """The single object perception produces and the planner consumes."""

    url: str
    title: str
    dom_summary: DomSummary
    screenshot_path: Optional[str] = None
    viewport: Optional[ViewportInfo] = None
    # Which perception mode produced the targeting info for this step.
    perception_mode: Literal["dom", "vision"] = "dom"
    perception_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Results / history
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    READ = "read"
    RISK = "risk"


class StepRecord(BaseModel):
    """One executed step, recorded for history, recipes and the run report."""

    index: int
    action: Action
    risk: RiskLevel
    ok: bool = True
    error: Optional[str] = None
    perception_mode: Literal["dom", "vision"] = "dom"
    url_before: Optional[str] = None
    url_after: Optional[str] = None
    screenshot_path: Optional[str] = None
    # Locator actually used to (re-)find the element, for recipe replay.
    used_locator: Optional[str] = None
    extracted: Optional[Any] = None


class RunResult(BaseModel):
    """Final result of a task run."""

    goal: str
    ok: bool
    steps: list[StepRecord] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    run_dir: Optional[str] = None
    message: Optional[str] = None
