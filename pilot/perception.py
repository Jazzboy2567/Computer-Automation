"""Perception — hybrid: DOM summary first, screenshot/vision as a fallback.

Default targeting/extraction uses the DOM summary (cheap, precise, stable). We
capture a screenshot every step regardless. We only *switch to vision mode* when
the DOM summary is empty or obfuscated — a canvas-rendered app, a cross-origin
iframe, or heavy JS that exposes no real elements. That condition is detected
explicitly and the chosen mode is logged on the `PageState`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .browser import Browser
from .schemas import PageState

log = logging.getLogger("pilot.perception")


async def _looks_obfuscated(browser: Browser, interactive_count: int) -> tuple[bool, str]:
    """Decide whether DOM targeting is viable for this page.

    Obfuscated == essentially nothing actionable in the DOM, typically because
    the page paints to <canvas>, lives in a cross-origin iframe, or builds its
    UI in a way the accessibility tree can't see.
    """
    assert browser.page
    if interactive_count > 0:
        return False, ""

    # No interactive elements at all — figure out why for the log.
    metrics = await browser.page.evaluate(
        """() => {
            const vw = window.innerWidth, vh = window.innerHeight;
            let canvasArea = 0;
            for (const c of document.querySelectorAll('canvas')) {
                const r = c.getBoundingClientRect();
                canvasArea += Math.max(0, r.width) * Math.max(0, r.height);
            }
            let xframe = 0;
            for (const f of document.querySelectorAll('iframe,frame')) {
                try { if (!f.contentDocument) xframe++; } catch (e) { xframe++; }
            }
            return {
                canvasFrac: canvasArea / (vw * vh),
                crossFrames: xframe,
                textLen: (document.body ? (document.body.innerText || '').length : 0),
            };
        }"""
    )
    if metrics["canvasFrac"] > 0.4:
        return True, "DOM has no actionable elements; large <canvas> detected"
    if metrics["crossFrames"] > 0:
        return True, "DOM has no actionable elements; cross-origin iframe present"
    return True, "DOM summary empty / no actionable elements"


async def perceive(
    browser: Browser,
    run_dir: Optional[Path] = None,
    step_index: int = 0,
    keywords: Optional[list[str]] = None,
) -> PageState:
    """Build the single page-state object the planner consumes."""
    summary = await browser.get_dom_summary(keywords=keywords)
    viewport = await browser.viewport_info()

    shot_path = None
    if run_dir is not None:
        shot_path = await browser.screenshot(Path(run_dir) / f"step_{step_index:02d}.png")

    interactive = sum(1 for e in summary.elements if e.kind == "interactive")
    obfuscated, why = await _looks_obfuscated(browser, interactive)

    mode = "vision" if obfuscated else "dom"
    note = None
    if obfuscated:
        note = f"vision fallback: {why}"
        log.warning("perception step %d -> VISION mode (%s)", step_index, why)
        summary.notes.append(note)
    else:
        log.info(
            "perception step %d -> DOM mode (%d elements, %d interactive)",
            step_index,
            len(summary.elements),
            interactive,
        )

    return PageState(
        url=summary.url,
        title=summary.title,
        dom_summary=summary,
        screenshot_path=shot_path,
        viewport=viewport,
        perception_mode=mode,
        perception_note=note,
    )
