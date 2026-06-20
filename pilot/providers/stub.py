"""StubProvider — scripted actions, zero API calls.

Used by the smoke tests and for offline development. You hand it a list of
actions (dicts or `Action`s) and it returns them in order, ignoring the page
state. When the script is exhausted it returns a `done` action so the loop ends
cleanly rather than hanging.
"""

from __future__ import annotations

from typing import Iterable

from ..schemas import Action, ActionType, PageState, StepRecord
from .base import Provider, parse_action


class StubProvider(Provider):
    name = "stub"

    def __init__(self, script: Iterable[dict | Action] | None = None):
        self._script: list[Action] = []
        for item in script or []:
            self._script.append(item if isinstance(item, Action) else parse_action(item))
        self._i = 0
        # Records what was actually asked of it (handy for assertions in tests).
        self.calls: list[Action] = []

    async def decide(
        self,
        goal: str,
        page_state: PageState,
        history: list[StepRecord],
    ) -> Action:
        if self._i < len(self._script):
            action = self._script[self._i]
            self._i += 1
        else:
            action = Action(type=ActionType.DONE, done=True, reasoning="script exhausted")
        self.calls.append(action)
        return action
