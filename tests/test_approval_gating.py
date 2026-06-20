"""Smoke test 4 — approval gating.

A `risk` action (clicking "Proceed to checkout") pauses for confirmation in
`checkpoint` mode and waits; `read` actions (extract) never prompt. Autonomous
mode never prompts; step mode prompts for everything.
"""

from __future__ import annotations

import pytest

from conftest import fixture_url
from pilot.agent import Agent, classify_risk, needs_confirmation
from pilot.config import ApprovalMode
from pilot.providers.base import Provider
from pilot.schemas import Action, ActionType, PageState, RiskLevel, StepRecord


class CheckoutProvider(Provider):
    """A tiny scripted 'model' that resolves the checkout ref from page state."""

    name = "checkout-test"

    def __init__(self):
        self.i = 0

    async def decide(self, goal, page_state: PageState, history: list[StepRecord]) -> Action:
        self.i += 1
        if self.i == 1:
            # read action: extract products
            return Action(type=ActionType.EXTRACT, selector="article.product_pod",
                          store_as="books", reasoning="read products")
        if self.i == 2:
            # risk action: click the checkout button (found by accessible name)
            ref = next(e.ref for e in page_state.dom_summary.elements
                       if e.name == "Proceed to checkout")
            return Action(type=ActionType.CLICK, ref=ref, reasoning="proceed to checkout")
        return Action(type=ActionType.DONE, done=True, reasoning="stop")


# --- unit-level checks of the classifier / gate -----------------------------

def test_risk_classifier():
    read = Action(type=ActionType.EXTRACT, selector="x", reasoning="r")
    assert classify_risk(read) is RiskLevel.READ

    checkout = Action(type=ActionType.CLICK, ref="e9", reasoning="r")
    assert classify_risk(checkout, target_name="Proceed to checkout") is RiskLevel.RISK
    assert classify_risk(checkout, target_name="A Light in the Attic") is RiskLevel.READ


def test_gate_rules():
    assert needs_confirmation(ApprovalMode.AUTONOMOUS, RiskLevel.RISK) is False
    assert needs_confirmation(ApprovalMode.CHECKPOINT, RiskLevel.READ) is False
    assert needs_confirmation(ApprovalMode.CHECKPOINT, RiskLevel.RISK) is True
    assert needs_confirmation(ApprovalMode.STEP, RiskLevel.READ) is True


# --- integration: checkpoint pauses on the risk action only -----------------

@pytest.mark.asyncio
async def test_checkpoint_pauses_only_on_risk(browser, settings):
    await browser.goto(fixture_url("books.html"))
    settings.approval_mode = ApprovalMode.CHECKPOINT

    prompted: list[tuple[str, str]] = []

    async def approval(action: Action, risk: RiskLevel) -> bool:
        prompted.append((action.short(), risk.value))
        return False  # decline -> aborts at the gate (never proceed past it)

    agent = Agent(browser, CheckoutProvider(), settings, approval=approval)
    result = await agent.run("Buy a book")

    # Exactly one prompt, and it was the checkout click (risk). The earlier
    # extract (read) ran without prompting.
    assert len(prompted) == 1
    assert prompted[0][1] == "risk"
    assert "click" in prompted[0][0]
    assert not result.ok
    assert "declined" in (result.message or "")
    # The read extraction still happened before the gate.
    assert "books" in result.extracted


@pytest.mark.asyncio
async def test_autonomous_never_prompts(browser, settings):
    await browser.goto(fixture_url("books.html"))
    settings.approval_mode = ApprovalMode.AUTONOMOUS

    prompted = []

    async def approval(action, risk):
        prompted.append(action.short())
        return True

    agent = Agent(browser, CheckoutProvider(), settings, approval=approval)
    await agent.run("Buy a book")
    assert prompted == []
