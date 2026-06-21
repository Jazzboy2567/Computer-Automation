"""Browser layer — Playwright wrapper + the `get_dom_summary` perception core.

Design notes
------------
* We use ``launch_persistent_context`` with a dedicated ``user-data-dir`` so YOUR
  logins/cookies persist between runs. You log in manually; Pilot never types,
  stores or handles a credential.
* ``get_dom_summary`` runs ONE injected JS pass (``dom_summary.js``) that returns
  a flat, document-ordered list of meaningful elements with accessibility data,
  visibility flags and a robust locator chain. Python then applies the token
  budget + keyword prioritization and keeps a ref->element map.
* ``click``/``type`` re-find an element by ref with a fast path (the in-page
  ``data-pilot-ref`` tag) and, if that is stale, the stored locator fallback
  chain — re-snapshotting as a last resort.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import (
    BrowserContext,
    Frame,
    Locator,
    Page,
    async_playwright,
)

from .config import Settings
from .schemas import DomElement, DomSummary, ViewportInfo

REF_ATTR = "data-pilot-ref"  # must match dom_summary.js

# Load the injected DOM-summary script once at import time.
_JS_PATH = Path(__file__).with_name("dom_summary.js")
_DOM_SUMMARY_JS = _JS_PATH.read_text(encoding="utf-8")


class StaleRefError(RuntimeError):
    """Raised when a ref cannot be re-found even after a fresh snapshot."""


def _approx_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


class Browser:
    """Async Playwright wrapper exposing the agent's browser primitives."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pw = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        # Current snapshot state: ref -> element, for re-finding.
        self._refs: dict[str, DomElement] = {}
        self._summary: Optional[DomSummary] = None

    # ------------------------------------------------------------------ life
    async def start(self) -> None:
        self.settings.profile_dir.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        # Persistent context => cookies/logins survive across runs.
        self.context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.settings.profile_dir),
            headless=not self.settings.headed,
            viewport={
                "width": self.settings.viewport_width,
                "height": self.settings.viewport_height,
            },
            # A normal-looking UA; we ship NO anti-bot evasion beyond this.
            args=["--disable-blink-features=AutomationControlled"],
        )
        self.page = (
            self.context.pages[0]
            if self.context.pages
            else await self.context.new_page()
        )

    async def close(self) -> None:
        try:
            if self.context:
                await self.context.close()
        finally:
            if self._pw:
                await self._pw.stop()

    # ------------------------------------------------------------- primitives
    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        assert self.page
        await self.page.goto(url, wait_until=wait_until)

    async def screenshot(self, path: str | Path, full_page: bool = False) -> str:
        assert self.page
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        await self.page.screenshot(path=str(path), full_page=full_page)
        return str(path)

    async def wait(
        self, seconds: float | None = None, wait_for: str | None = None
    ) -> None:
        assert self.page
        if wait_for in ("load", "domcontentloaded", "networkidle"):
            await self.page.wait_for_load_state(wait_for)  # type: ignore[arg-type]
        elif wait_for:
            await self.page.wait_for_selector(wait_for)
        if seconds:
            await self.page.wait_for_timeout(seconds * 1000)

    async def scroll(
        self,
        direction: str = "down",
        amount: int | None = None,
        ref: str | None = None,
    ) -> None:
        assert self.page
        if ref:
            loc = await self._find(ref)
            await loc.scroll_into_view_if_needed()
            return
        if direction == "top":
            await self.page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            dy = amount or self.settings.viewport_height * 0.8
            dy = -dy if direction == "up" else dy
            await self.page.evaluate(f"window.scrollBy(0, {int(dy)})")

    async def viewport_info(self) -> ViewportInfo:
        assert self.page
        info = await self.page.evaluate(
            """() => ({
                width: window.innerWidth, height: window.innerHeight,
                scroll_x: window.scrollX, scroll_y: window.scrollY,
                page_height: Math.max(document.body ? document.body.scrollHeight : 0,
                    document.documentElement ? document.documentElement.scrollHeight : 0)
            })"""
        )
        return ViewportInfo(**info)

    # --------------------------------------------------------- DOM SUMMARY
    async def get_dom_summary(
        self,
        keywords: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
        in_viewport_only: bool = False,
    ) -> DomSummary:
        """Run the injected pass, then apply the token budget + prioritization.

        Refs are assigned IN the JS pass in document order, so they are stable
        for a given page state regardless of how budgeting trims the output.
        """
        assert self.page
        budget = max_tokens or self.settings.dom_token_budget
        raw = await self.page.evaluate(_DOM_SUMMARY_JS, {"maxName": 120})

        all_elems = [DomElement(**n) for n in raw["nodes"]]
        if in_viewport_only:
            all_elems = [e for e in all_elems if e.in_viewport]

        kept, truncated = self._apply_budget(all_elems, budget, keywords)

        summary = DomSummary(
            url=raw["url"],
            title=raw["title"],
            elements=kept,
            truncated=truncated,
            total_found=len(all_elems),
            notes=list(raw.get("notes", [])),
        )
        summary.approx_tokens = _approx_tokens(summary.render())

        # Track full ref map (ALL elements, even budget-trimmed ones) so a model
        # that scrolls/asks for a region can still target trimmed refs.
        self._refs = {e.ref: e for e in all_elems}
        self._summary = summary
        return summary

    def _apply_budget(
        self,
        elems: list[DomElement],
        budget: int,
        keywords: Optional[list[str]],
    ) -> tuple[list[DomElement], bool]:
        """Greedily keep the highest-value elements within the token budget.

        Priority: interactive > keyword-matching > in-viewport. We select by
        score, then RENDER in document order so the page still reads top-to-bottom.
        """
        kws = [k.lower() for k in (keywords or [])]

        def score(idx: int, e: DomElement) -> float:
            s = 0.0
            if e.kind == "interactive":
                s += 100
            if e.in_viewport:
                s += 10
            if kws:
                hay = (e.name + " " + " ".join(e.attrs.values())).lower()
                s += 50 * sum(1 for k in kws if k in hay)
            # Slight preference for earlier elements to break ties deterministically.
            s -= idx * 0.001
            return s

        indexed = list(enumerate(elems))
        order = sorted(indexed, key=lambda p: score(p[0], p[1]), reverse=True)

        kept_idx: set[int] = set()
        used = 0
        for idx, e in order:
            cost = _approx_tokens(e.to_line())
            if used + cost > budget and kept_idx:
                continue
            kept_idx.add(idx)
            used += cost
            if used >= budget:
                break

        kept = [e for i, e in indexed if i in kept_idx]
        return kept, len(kept) < len(elems)

    # ------------------------------------------------------ ref re-finding
    def _frames(self) -> list[Frame]:
        assert self.page
        return self.page.frames

    async def _resolve_locator(self, frame: Frame, loc: str) -> Optional[Locator]:
        """Turn one scheme string from the fallback chain into a Locator."""
        try:
            if loc.startswith("css="):
                return frame.locator(loc[4:])
            if loc.startswith("xpath="):
                return frame.locator(loc)
            if loc.startswith("text="):
                return frame.get_by_text(loc[5:], exact=False).first
            if loc.startswith("role="):
                body = loc[5:]
                role, _, rest = body.partition("|")
                name = ""
                if rest.startswith("name="):
                    name = rest[5:]
                return frame.get_by_role(role, name=name, exact=False).first  # type: ignore[arg-type]
        except Exception:
            return None
        return None

    async def _find(self, ref: str, _retry: bool = True) -> Locator:
        """Re-find an element by ref.

        1. Fast path: the in-page ``data-pilot-ref`` tag (current snapshot).
        2. Fallback: the stored robust locator chain (survives minor DOM churn).
        3. Last resort: re-snapshot once, then retry the fast path.
        """
        assert self.page
        sel = f'[{REF_ATTR}="{ref}"]'
        for frame in self._frames():
            try:
                loc = frame.locator(sel)
                if await loc.count() >= 1:
                    return loc.first
            except Exception:
                continue

        # Fallback chain.
        elem = self._refs.get(ref)
        if elem:
            for frame in self._frames():
                for cand in elem.locators:
                    resolved = await self._resolve_locator(frame, cand)
                    if resolved is None:
                        continue
                    try:
                        if await resolved.count() >= 1:
                            return resolved.first
                    except Exception:
                        continue

        # Last resort: a fresh snapshot may re-tag a churned DOM.
        if _retry:
            await self.get_dom_summary()
            return await self._find(ref, _retry=False)

        raise StaleRefError(f"Could not locate ref {ref!r} after re-snapshot")

    async def locator_used_for(self, ref: str) -> Optional[str]:
        """Best durable locator for `ref` (for recording into a recipe)."""
        elem = self._refs.get(ref)
        if elem and elem.locators:
            return elem.locators[0]
        return None

    # --------------------------------------------------------- interactions
    async def click(self, ref: str) -> str:
        loc = await self._find(ref)
        await loc.scroll_into_view_if_needed()
        await loc.click()
        return (await self.locator_used_for(ref)) or ref

    async def type(self, ref: str, text: str, clear: bool = True) -> str:
        loc = await self._find(ref)
        await loc.scroll_into_view_if_needed()
        try:
            # fill clears + sets in one go; best for standard inputs.
            await loc.fill(text)
        except Exception:
            await loc.click()
            if clear:
                await loc.press("Control+A")
                await loc.press("Delete")
            await loc.press_sequentially(text)
        return (await self.locator_used_for(ref)) or ref

    # ----------------------------------------- act by durable locator (replay)
    async def _find_by_locator(self, loc_str: str) -> Locator:
        """Resolve a stored locator string across all frames (for replay)."""
        for frame in self._frames():
            resolved = await self._resolve_locator(frame, loc_str)
            if resolved is None:
                continue
            try:
                if await resolved.count() >= 1:
                    return resolved.first
            except Exception:
                continue
        raise StaleRefError(f"Locator did not match during replay: {loc_str!r}")

    async def click_locator(self, loc_str: str) -> None:
        loc = await self._find_by_locator(loc_str)
        await loc.scroll_into_view_if_needed()
        await loc.click()

    async def type_locator(self, loc_str: str, text: str) -> None:
        loc = await self._find_by_locator(loc_str)
        await loc.scroll_into_view_if_needed()
        try:
            await loc.fill(text)
        except Exception:
            await loc.click()
            await loc.press_sequentially(text)

    # ----------------------------------------------------------- extraction
    async def extract(
        self,
        selector: str,
        fields: Optional[dict[str, str]] = None,
    ) -> Any:
        """Pull text/data out of the page with plain selectors.

        * No ``fields``: returns a list of trimmed innerText for each match.
        * With ``fields``: ``{name: "relative-selector"}`` (optionally
          ``"selector@attr"`` to read an attribute, e.g. ``"a@href"``) is applied
          relative to each ``selector`` match, returning a list of dicts.
        """
        assert self.page
        containers = self.page.locator(selector)
        n = await containers.count()
        if not fields:
            return [
                (await containers.nth(i).inner_text()).strip() for i in range(n)
            ]

        rows: list[dict[str, Any]] = []
        for i in range(n):
            c = containers.nth(i)
            row: dict[str, Any] = {}
            for name, spec in fields.items():
                sub, _, attr = spec.partition("@")
                try:
                    target = c.locator(sub) if sub else c
                    if attr:
                        row[name] = await target.first.get_attribute(attr)
                    else:
                        row[name] = (await target.first.inner_text()).strip()
                except Exception:
                    row[name] = None
            rows.append(row)
        return rows

    # ------------------------------------------------------------- vision aid
    async def click_xy(self, x: int, y: int) -> None:
        """Vision-fallback click by absolute coordinate."""
        assert self.page
        await self.page.mouse.click(x, y)

    @property
    def current_summary(self) -> Optional[DomSummary]:
        return self._summary
