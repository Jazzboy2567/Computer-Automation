"""The seam for the REAL Steam game.

The simulated env validates the learner; to play an actual game you provide
three game-specific pieces and compose them into a `ScreenGameEnv`:

* `Capturer.grab()`        -> a screenshot of the game (via the computer-use MCP).
* `FeatureExtractor.extract(frame)` -> the structured observation (player health,
  enemies nearby, ...) — the "consistent important data" read from the frame.
* `ActionDriver.do(action)` -> perform the key/mouse input for an action.

Nothing here calls the OS by itself — the host wires in the computer-use
functions at runtime (after `request_access`), which keeps this module testable
and free of hard desktop dependencies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from .env import GameEnv, Observation


class Capturer(ABC):
    @abstractmethod
    def grab(self) -> Any:
        """Return a screenshot (path / bytes / image object)."""


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, frame: Any) -> Observation:
        """Turn a screenshot into the structured observation the agent uses."""


class ActionDriver(ABC):
    @abstractmethod
    def do(self, action: str) -> None:
        """Perform the key/mouse input for an action."""


class ComputerUseCapturer(Capturer):
    """Screenshots via the computer-use MCP, injected by the host at runtime."""

    def __init__(self, screenshot_fn: Callable[[], Any] | None = None):
        self._fn = screenshot_fn

    def grab(self) -> Any:
        if self._fn is None:
            raise NotImplementedError(
                "Wire a screenshot function from the computer-use MCP "
                "(request_access -> screenshot) into ComputerUseCapturer."
            )
        return self._fn()


class ScreenGameEnv(GameEnv):
    """A `GameEnv` backed by a live game: capture -> extract -> act.

    Drop-in compatible with the training loop, so once you supply a real
    `Capturer`/`FeatureExtractor`/`ActionDriver` for a Steam game, the exact same
    agent + reward + `train()` apply.
    """

    def __init__(
        self,
        capturer: Capturer,
        extractor: FeatureExtractor,
        driver: ActionDriver,
        action_space: list[str],
        done_fn: Callable[[Observation], bool],
        settle: float = 0.0,
    ):
        self.capturer = capturer
        self.extractor = extractor
        self.driver = driver
        self.action_space = list(action_space)
        self._done_fn = done_fn
        self.settle = settle

    def observation_fields(self) -> list[str]:
        return []

    def reset(self) -> Observation:
        return self.extractor.extract(self.capturer.grab())

    def step(self, action: str) -> tuple[Observation, bool, dict[str, Any]]:
        self.driver.do(action)
        if self.settle:
            time.sleep(self.settle)
        obs = self.extractor.extract(self.capturer.grab())
        return obs, bool(self._done_fn(obs)), {}
