"""Pilot — a personal, local browser-automation agent.

Pilot drives YOUR browser, logged into YOUR own accounts, and performs tasks the
way you would by hand: navigating, reading pages, extracting information and
compiling results. It is a general-purpose framework ("personal Browser Use"),
not a site-specific scraper.

Module map
----------
- ``pilot.browser``      Playwright wrapper + the ``get_dom_summary`` perception core.
- ``pilot.perception``   Hybrid perception (DOM summary first, vision fallback).
- ``pilot.providers``    Swappable model providers (Stub / Anthropic / OpenAI).
- ``pilot.agent``        Perceive -> decide -> (confirm) -> act loop + approval modes.
- ``pilot.recipes``      Record once, replay deterministically with no model calls.
- ``pilot.tasks``        Task definitions + plain-code comparison/ranking.
- ``pilot.output``       Markdown / JSON / CSV reporters + run artifacts.
- ``pilot.server``       FastAPI app + small web UI.
"""

__version__ = "0.1.0"
