"""Server smoke test — the responsible-use gate + read-only endpoints.

Uses FastAPI's TestClient (no browser launched). Verifies that a run cannot
start until the first-run notice is acknowledged, and that the task/recipe
listings render.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pilot import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate the acknowledgement marker so the test never touches profiles/.
    monkeypatch.setattr("pilot.config.ACK_FILE", tmp_path / ".acknowledged")
    return TestClient(server.app)


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Pilot" in r.text


def test_start_blocked_until_acknowledged(client):
    # Not acknowledged yet.
    assert client.get("/api/ack").json() == {"acknowledged": False}

    blocked = client.post("/api/start", json={"goal": "do something", "provider": "stub"})
    assert blocked.status_code == 403

    # Acknowledge, then the gate opens (we don't actually start a browser here).
    client.post("/api/ack", json={"accept": True})
    assert client.get("/api/ack").json() == {"acknowledged": True}


def test_listing_endpoints(client):
    tasks = client.get("/api/tasks")
    assert tasks.status_code == 200
    assert "tasks" in tasks.json()

    recipes = client.get("/api/recipes")
    assert recipes.status_code == 200
    assert "recipes" in recipes.json()
