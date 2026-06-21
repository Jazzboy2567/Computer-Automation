"""LocalProvider tests — no Ollama server required.

Verifies the provider is registered and that an unreachable server yields a
clear, actionable RuntimeError (rather than a raw httpx traceback).
"""

from __future__ import annotations

import pytest

from pilot.providers import get_provider
from pilot.providers.local_provider import LocalProvider
from pilot.schemas import DomSummary, PageState


def _page_state() -> PageState:
    summary = DomSummary(url="about:blank", title="t")
    return PageState(url="about:blank", title="t", dom_summary=summary)


def test_local_provider_registered():
    p = get_provider("local", host="http://127.0.0.1:9", model="nope")
    assert isinstance(p, LocalProvider)
    assert p.name == "local"
    assert p.host == "http://127.0.0.1:9"


@pytest.mark.asyncio
async def test_unreachable_server_raises_clear_error():
    # Port 9 (discard) refuses fast -> connection error wrapped as RuntimeError.
    p = LocalProvider(host="http://127.0.0.1:9", model="nope")
    with pytest.raises(RuntimeError, match="Ollama"):
        await p.decide("do something", _page_state(), [])
