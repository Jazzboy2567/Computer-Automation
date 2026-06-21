"""LocalProvider — a fully-local model via Ollama (no cloud, no API key).

Talks to a local Ollama server over HTTP (uses the already-bundled ``httpx``),
so there is no extra dependency and nothing leaves your machine. It reuses the
shared action schema/prompt; output is requested as JSON and parsed into an
`Action`. In vision mode the step screenshot is attached for vision-capable
local models.

Setup:  install Ollama (https://ollama.com), then e.g. ``ollama pull llama3.1``.
Config: ``OLLAMA_HOST`` (default http://localhost:11434), ``OLLAMA_MODEL``
(default ``llama3.1``).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx

from ..schemas import Action, PageState, StepRecord
from .base import ACTION_GUIDE, Provider, build_user_text, parse_action

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"

# Local models don't reliably support tool-use, so we ask for a single JSON
# object and parse it. This appendix names the exact keys to emit.
_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a single JSON object (no prose, no code fences) using "
    "these keys as needed: type (one of goto|click|type|scroll|extract|wait|done), "
    "ref, url, text, selector, fields, store_as, direction, amount, seconds, "
    "wait_for, x, y, done, reasoning. Include only the keys relevant to the action."
)


class LocalProvider(Provider):
    name = "local"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout

    async def decide(
        self,
        goal: str,
        page_state: PageState,
        history: list[StepRecord],
    ) -> Action:
        user_msg: dict = {"role": "user", "content": build_user_text(goal, page_state, history)}
        if page_state.perception_mode == "vision" and page_state.screenshot_path:
            img = Path(page_state.screenshot_path)
            if img.exists():
                # Ollama takes base64 images on the message (vision-capable models).
                user_msg["images"] = [base64.b64encode(img.read_bytes()).decode()]

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": ACTION_GUIDE + _JSON_INSTRUCTION},
                user_msg,
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Local Ollama request to {self.host} failed "
                f"({type(e).__name__}: {e}). Is Ollama running and is the model "
                f"{self.model!r} pulled? See https://ollama.com"
            ) from e

        content = (data.get("message") or {}).get("content", "")
        return parse_action(content)
