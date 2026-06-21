"""Swappable model providers.

Every provider implements ONE core method, ``decide(goal, page_state, history)
-> Action``, returning the same centralized `Action` schema. Prompts differ only
in formatting; the action vocabulary is shared (see ``pilot.providers.base``).
"""

from __future__ import annotations

from .base import Provider
from .stub import StubProvider


def get_provider(name: str, **kwargs) -> Provider:
    """Factory. ``name`` in {"stub", "anthropic", "openai"}.

    Real providers are imported lazily so the core (and the smoke tests) never
    require the ``anthropic``/``openai`` SDKs to be installed.
    """
    name = (name or "stub").lower()
    if name == "stub":
        return StubProvider(**kwargs)
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name!r}")


__all__ = ["Provider", "StubProvider", "get_provider"]
