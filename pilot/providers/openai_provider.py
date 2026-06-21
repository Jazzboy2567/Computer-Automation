"""OpenAIProvider — alternate real provider via function calling.

Mirrors `AnthropicProvider`: same shared action schema, structured output via a
forced tool call, key read from ``OPENAI_API_KEY`` at call time (never stored).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from ..config import env_api_key
from ..schemas import Action, PageState, StepRecord
from .base import ACTION_GUIDE, ACTION_JSON_SCHEMA, Provider, build_user_text, parse_action

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "openai SDK not installed. `pip install openai` "
                "(or `pip install -e .[openai]`)."
            ) from e
        key = env_api_key("openai")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    async def decide(
        self,
        goal: str,
        page_state: PageState,
        history: list[StepRecord],
    ) -> Action:
        user_content: list[dict] = [
            {"type": "text", "text": build_user_text(goal, page_state, history)}
        ]
        if page_state.perception_mode == "vision" and page_state.screenshot_path:
            img = Path(page_state.screenshot_path)
            if img.exists():
                b64 = base64.b64encode(img.read_bytes()).decode()
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": ACTION_GUIDE},
                {"role": "user", "content": user_content},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "act",
                        "description": "Perform the single next browser action.",
                        "parameters": ACTION_JSON_SCHEMA,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "act"}},
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            return parse_action(json.loads(msg.tool_calls[0].function.arguments))
        return parse_action(msg.content or "")
