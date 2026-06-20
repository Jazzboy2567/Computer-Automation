"""Provider base class + the centralized action schema/prompt shared by all.

The action JSON shape is defined ONCE here (both as prose for the prompt and as
a JSON schema for tool-use / function-calling). Concrete providers reuse these
so they only differ in how they talk to their API.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schemas import Action, PageState, StepRecord

# ---------------------------------------------------------------------------
# Shared action schema (single source of truth for every provider)
# ---------------------------------------------------------------------------

# JSON schema for the one "act" tool / function all real providers expose.
ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["goto", "click", "type", "scroll", "extract", "wait", "done"],
            "description": "The single next action to perform.",
        },
        "ref": {"type": "string", "description": "Element ref id from the DOM summary, e.g. 'e12'."},
        "url": {"type": "string", "description": "Target URL (for type=goto)."},
        "text": {"type": "string", "description": "Text to type (for type=type)."},
        "selector": {"type": "string", "description": "CSS selector (for type=extract)."},
        "fields": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Relative field map for extract, e.g. {'title':'h3 a','price':'.price'}. Use 'sel@attr' for attributes, e.g. 'a@href'.",
        },
        "store_as": {"type": "string", "description": "Name to store extracted data under."},
        "direction": {"type": "string", "enum": ["up", "down", "top", "bottom", "to_element"]},
        "amount": {"type": "integer", "description": "Scroll distance in pixels."},
        "seconds": {"type": "number", "description": "Seconds to wait (for type=wait)."},
        "wait_for": {"type": "string", "description": "Selector or load state to wait for."},
        "x": {"type": "integer", "description": "Vision fallback click X (only in vision mode)."},
        "y": {"type": "integer", "description": "Vision fallback click Y (only in vision mode)."},
        "done": {"type": "boolean", "description": "True when the goal is complete."},
        "reasoning": {"type": "string", "description": "One short sentence explaining this step."},
    },
    "required": ["type", "reasoning"],
}

ACTION_GUIDE = """\
You control a real web browser to accomplish the user's GOAL. Each turn you see
the current page as a compact DOM summary: one line per element, e.g.
  e12 [button] "Add to Cart"  (visible)
  e13 [link] "Sony WH-1000XM5"  href=/p/123  (needs-scroll)
  e14 [text] "$348.00"
Refs (e12, e13, ...) are how you target elements. "needs-scroll" means you must
scroll before the element is in view.

Respond with EXACTLY ONE next action using the provided schema:
- goto: navigate (set url)
- click: click an element (set ref)
- type: type into a field (set ref + text)
- scroll: reveal more of the page (set direction; 'down' by default)
- extract: pull data with CSS (set selector; optional fields map + store_as)
- wait: pause for content (set seconds or wait_for)
- done: the goal is achieved (set done=true)

Rules:
- Prefer DOM refs over coordinates. Only use x/y when told you are in VISION mode.
- Extract data with `extract` rather than reading values off screenshots.
- Set done=true as soon as the goal is satisfied. Don't loop pointlessly.
- Never attempt to solve CAPTCHAs or bot checks — stop and ask the human instead.
"""


def format_history(history: list[StepRecord], limit: int = 8) -> str:
    if not history:
        return "(no actions yet)"
    lines = []
    for s in history[-limit:]:
        status = "ok" if s.ok else f"ERROR: {s.error}"
        extra = ""
        if s.extracted is not None:
            extra = f" -> stored {s.action.store_as or 'data'}"
        lines.append(f"{s.index}. {s.action.short()} [{status}]{extra}")
    return "\n".join(lines)


def build_user_text(goal: str, page_state: PageState, history: list[StepRecord]) -> str:
    """The shared textual context every provider sends (mode-aware)."""
    parts = [f"GOAL: {goal}", "", "RECENT ACTIONS:", format_history(history), ""]
    if page_state.perception_mode == "vision":
        parts.append(
            "PERCEPTION MODE: VISION — the DOM summary is empty/obfuscated for "
            "this page. Use the attached screenshot and target with x/y "
            "coordinates, or navigate elsewhere."
        )
        if page_state.perception_note:
            parts.append(f"({page_state.perception_note})")
    else:
        parts.append("PERCEPTION MODE: DOM")
        parts.append("")
        parts.append(page_state.dom_summary.render())
    parts.append("")
    parts.append("Decide the single next action.")
    return "\n".join(parts)


def parse_action(data: dict[str, Any] | str) -> Action:
    """Coerce a model's JSON (string or dict) into a validated `Action`."""
    if isinstance(data, str):
        data = _extract_json(data)
    # Mirror an explicit DONE type onto the done flag and vice-versa.
    if data.get("type") == "done":
        data["done"] = True
    return Action.model_validate(data)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a free-text model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class Provider(ABC):
    """All model providers implement this single decision method."""

    name: str = "base"

    @abstractmethod
    async def decide(
        self,
        goal: str,
        page_state: PageState,
        history: list[StepRecord],
    ) -> Action:
        """Return the next `Action` to perform, given the current page state."""
        raise NotImplementedError
