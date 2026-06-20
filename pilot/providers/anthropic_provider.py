"""AnthropicProvider — the default real provider (latest Claude).

Uses tool-use to get structured output matching the shared action schema. The
API key is read from ``ANTHROPIC_API_KEY`` at call time and never stored. In
vision mode the step screenshot is attached as an image block.
"""

from __future__ import annotations

import base64
from pathlib import Path

from ..config import env_api_key
from ..schemas import Action, PageState, StepRecord
from .base import ACTION_GUIDE, ACTION_JSON_SCHEMA, Provider, build_user_text, parse_action

# Latest Claude model id (see project environment notes).
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover - exercised only with extra installed
            raise RuntimeError(
                "anthropic SDK not installed. `pip install anthropic` "
                "(or `pip install -e .[anthropic]`)."
            ) from e
        key = env_api_key("anthropic")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        self._client = AsyncAnthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    async def decide(
        self,
        goal: str,
        page_state: PageState,
        history: list[StepRecord],
    ) -> Action:
        content: list[dict] = [{"type": "text", "text": build_user_text(goal, page_state, history)}]
        if page_state.perception_mode == "vision" and page_state.screenshot_path:
            img = Path(page_state.screenshot_path)
            if img.exists():
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(img.read_bytes()).decode(),
                        },
                    }
                )

        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=ACTION_GUIDE,
            tools=[
                {
                    "name": "act",
                    "description": "Perform the single next browser action.",
                    "input_schema": ACTION_JSON_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "act"},
            messages=[{"role": "user", "content": content}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return parse_action(dict(block.input))
        # Fallback: parse any text the model returned.
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_action(text)
