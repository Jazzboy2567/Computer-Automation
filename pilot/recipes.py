"""Recipes — record once, replay deterministically with NO model calls.

The first successful run of a task records its concrete steps (URLs, durable
locators, actions) into a JSON recipe. Later runs REPLAY those steps directly,
so the common path is fast, free and rate-limit-proof. The model is invoked
ONLY when a replayed step fails (locator missing / layout changed); then the
agent re-plans from the live page and the recipe can be re-recorded.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .config import RECIPES_DIR
from .schemas import Action, ActionType, RiskLevel, StepRecord


class RecipeStep(BaseModel):
    """One replayable step: the action plus the durable locator to re-find it."""

    action: Action
    locator: Optional[str] = None   # durable locator for click/type
    url_before: Optional[str] = None
    # Recorded risk so replay can gate (e.g. pause on checkout) without
    # re-perceiving the page.
    risk: RiskLevel = RiskLevel.READ


class Recipe(BaseModel):
    name: str
    goal: str
    sites: list[str] = Field(default_factory=list)
    created: str = ""
    provider: str = ""
    steps: list[RecipeStep] = Field(default_factory=list)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "recipe"


def build_recipe(
    name: str,
    goal: str,
    sites: list[str],
    history: list[StepRecord],
    provider: str = "",
) -> Recipe:
    """Convert a successful run's history into a replayable recipe.

    We keep only the steps that change page state or extract data. Screenshots,
    waits-only and the perception bookkeeping are dropped; click/type steps carry
    the durable locator actually used so replay needs no DOM snapshot/model.
    """
    steps: list[RecipeStep] = []
    for rec in history:
        if not rec.ok:
            continue
        a = rec.action
        if a.type in (ActionType.SCREENSHOT,):
            continue
        steps.append(
            RecipeStep(
                action=a,
                locator=rec.used_locator if a.type in (ActionType.CLICK, ActionType.TYPE) else None,
                url_before=rec.url_before,
                risk=rec.risk,
            )
        )
    return Recipe(
        name=name,
        goal=goal,
        sites=sites,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        provider=provider,
        steps=steps,
    )


class RecipeStore:
    """Tiny on-disk store: one JSON file per recipe under ``recipes/``."""

    def __init__(self, directory: Path = RECIPES_DIR):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.json"

    def save(self, recipe: Recipe) -> Path:
        p = self.path_for(recipe.name)
        p.write_text(recipe.model_dump_json(indent=2), encoding="utf-8")
        return p

    def load(self, name: str) -> Optional[Recipe]:
        p = self.path_for(name)
        if not p.exists():
            return None
        return Recipe.model_validate_json(p.read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))
