# ✈️ Pilot — a personal browser-automation agent

Pilot is a local tool that performs tasks on websites **the way you would by
hand**: navigating, reading pages, extracting information and compiling results.
It runs in **your** browser session, logged into **your** own accounts, and acts
on your behalf — think "personal Browser Use." It is a general-purpose
framework, **not** a site-specific scraper and **not** a tool for bypassing any
particular site.

> ⚠️ **Read the [Legal & responsible use](#legal--responsible-use) section before
> you run anything.** Automated access can violate some sites' Terms of Service
> and put your accounts at risk. You decide which sites you target, and you own
> that decision.

---

## Highlights

- **ML is the foreground** — you state a goal and a machine-learning engine
  produces the result (with performance metrics). The LLM (Ollama) is a
  **background planner** that only interprets the goal; a no-AI heuristic
  fallback means ML always runs. Each goal lives in its own isolated workspace.
- **Reliable perception** — `get_dom_summary` turns a live page into a compact,
  stable, token-efficient list of actionable + readable elements the model
  targets by short ref IDs (`e12`). DOM-first, with an automatic **vision
  fallback** for canvas/obfuscated pages.
- **No credentials, ever** — you log in manually in a persistent browser
  profile. Pilot never types, stores or handles a password.
- **Approval modes** — `read` actions run freely; `risk` actions (spending
  money, submitting, deleting, changing settings) pause for your approval.
- **Recipes** — the first successful run is recorded; later runs **replay
  deterministically with zero model calls** (fast, free, rate-limit-proof).
- **Swappable providers** — Stub (no API), Anthropic (default), OpenAI, and
  **local (Ollama)** for a fully-local, no-API-key model.
- **Plain-code comparison** — ranking/sorting is done in code, not by a model.
- **Web UI + CLI** — drive it and watch it work, or script it from a terminal.

---

## Quick start

### 1. Install

**With `uv` (recommended):**

```bash
uv venv
uv pip install -e ".[dev]"          # add ",anthropic,openai" for real providers
uv run playwright install chromium
```

**With venv + pip:**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"   # add ".[dev,anthropic,openai]" for real providers
python -m playwright install chromium
```

> Python 3.11+ is required. On Windows, the `make` targets work under Git Bash;
> otherwise run the underlying commands directly.

### 2. Verify (no network, no API key)

```bash
make test            # the five headless smoke tests
make demo            # full pipeline against a bundled fixture -> writes a report
# or: python -m pilot.cli demo
```

### 3. Launch the UI

```bash
make serve           # or: python -m pilot.cli serve  /  uvicorn pilot.server:app
```

Open <http://127.0.0.1:8000>, accept the first-run responsible-use notice, pick a
task or type an ad-hoc goal, and press **Start**.

---

## ML mode — the foreground

You state a goal; an ML engine produces the result. The LLM (Ollama) only plans
the task in the background, and is optional.

```bash
# No data + no Ollama: a heuristic planner + a bundled sample still produce a result
pilot ml "classify iris species" --planner heuristic

# Your own data; the engine trains + evaluates and writes a report
pilot ml "predict churn" --data customers.csv --target churned

# Let the local LLM interpret a fuzzy goal (falls back to heuristic if Ollama is down)
pilot ml "group these customers into segments" --data customers.csv --planner auto
```

How it works (`pilot/ml/`):

1. **Workspace** — each goal gets an isolated `ml_workspaces/<goal>_<ts>/` with a
   data snapshot, the model artifact, predictions, and a `report.md`.
2. **Planner (background AI)** — `OllamaPlanner` (local LLM, `format=json`) or the
   no-AI `HeuristicPlanner` turns the goal + a data profile into an `MLTaskSpec`
   (task type, target, model, metric). `--planner auto` tries Ollama, then falls
   back to the heuristic — so it runs with no AI at all.
3. **Engine (foreground ML)** — scikit-learn trains and evaluates the model and
   produces the result + metrics. **No LLM is in this path.** Supports
   **classification**, **regression**, and **clustering** today.
4. **Report** — `metrics.json` + a `report.md` with the chosen spec, a metrics
   table, and feature importances (or cluster sizes).

Configure the background model with `OLLAMA_MODEL` / `OLLAMA_HOST`. The model is
never what produces your result — that's always the ML engine.

## One-time manual login (the only way Pilot touches your accounts)

Pilot uses a **persistent browser profile** at `./profiles/default`, so cookies
and logins survive between runs. You log in **yourself**, once:

```bash
python -m pilot.cli run tasks/price_compare_books.json   # opens a real browser…
```

…or just launch any headed run and, while the browser window is open, navigate to
the site and sign in normally. The session is saved in the profile. **Pilot has
no login/credential code** — it cannot and will not enter a password for you. The
`profiles/` directory is git-ignored; never commit it.

---

## Defining a task

A **task** is a JSON file in `tasks/`: a goal, target sites, an output schema,
and an optional sort. Example (`tasks/price_compare_books.json`, runs fully
offline via the Stub provider):

```jsonc
{
  "name": "Price-compare books (demo)",
  "goal": "Collect every book's title, price and stock, then rank cheapest first.",
  "sites": ["https://books.toscrape.com/"],
  "output_schema": ["title", "price", "in_stock", "url"],
  "sort": ["price:asc"],          // also: "in_stock:desc", "match:desc"
  "provider": "stub",             // "anthropic" | "openai" | "stub"
  "recipe": "books-demo",         // record/replay under recipes/
  "script": [ /* optional scripted actions for a deterministic Stub run */ ]
}
```

Run it:

```bash
python -m pilot.cli run tasks/price_compare_books.json
python -m pilot.cli run tasks/jobs_gather.json --provider anthropic --approval checkpoint
```

Ranking is **plain code** (see `pilot/tasks.py`): `price:asc` parses currency,
`in_stock:desc` puts available items first, `match:desc` ranks by how many of the
task's `keywords` appear. The model is only used to drive perception/decisions —
never the ranking.

Three templates ship in `tasks/`: **price-compare**, **Steam wishlist price
check**, and **job-listing gather + rank**. The Steam/jobs examples need a
provider API key (and, for Steam, a manual login).

---

## How `get_dom_summary` works (the core of reliability)

`get_dom_summary` (in `pilot/browser.py` + the injected `pilot/dom_summary.js`)
runs **one** `page.evaluate` pass and produces lines like:

```
PAGE: All products
URL: https://books.toscrape.com/
e12 [link]   "Sapiens: A Brief History of Humankind"  href=/p/123  (visible)
e13 [button] "Add to basket"  (needs-scroll)
e14 [text]   "£54.23"  (visible)
```

- **Backbone:** the accessibility model — each kept node gets its ARIA **role**
  and **accessible name** computed in-page the way the a11y tree does
  (`aria-labelledby`/`aria-label`/associated `<label>`/`alt`/text/`title`).
- **Keeps only meaningful nodes:** interactive elements (links, buttons, inputs,
  selects, `[role=…]`, `[onclick]`, `contenteditable`, …) and text-bearing
  content (headings, list items, table cells, **prices**, labels). Pure layout
  wrappers are dropped.
- **Visibility filter:** excludes `display:none`, `visibility:hidden`,
  zero-size, `aria-hidden`, off-document nodes; flags each kept element
  **`visible`** vs **`needs-scroll`**.
- **Stable refs:** document-order traversal ⇒ the same page yields the same
  `e1, e2, …` across calls within a run, so recipes can rely on them.
- **Robust locators:** every element carries a fallback chain —
  `data-testid`/`id` → ARIA role+name → unique text → generated CSS path — so
  `click(ref)`/`type(ref)` re-find it after minor DOM churn. Elements are also
  tagged with `data-pilot-ref` for a fast re-find; if a ref goes stale, Pilot
  falls back to the locator chain and finally re-snapshots.
- **Token budget:** the summary is capped (~4k tokens by default). When over
  budget it prioritizes interactive elements and elements near the task's
  keywords; scroll and re-snapshot to reveal more.
- **iframes / shadow DOM:** same-origin iframes and open shadow roots are
  traversed; cross-origin/canvas content is noted so perception can fall back to
  vision.

**Hybrid perception** (`pilot/perception.py`): every step captures a screenshot
and builds a page state `{ url, title, dom_summary, screenshot, viewport_info }`.
If the DOM summary is empty/obfuscated (a `<canvas>` app, a cross-origin iframe),
perception switches to **vision mode** — the screenshot is sent to the model and
it targets by coordinates — and logs which mode was used.

---

## Approval modes & the risk classifier

Every action is tagged **`read`** or **`risk`** (`pilot/agent.py`):

- **`read`** — navigate, scroll, screenshot, extract, type into a field.
- **`risk`** — a click whose target's name implies spending money, submitting,
  sending/posting, deleting, or changing account settings (e.g. *Checkout*,
  *Place order*, *Delete*, *Pay*).

Gating is on the **action type**, not a counter. Three UI-toggleable modes:

| Mode | Behavior |
| --- | --- |
| `autonomous` | Run through; stop only on errors. |
| `checkpoint` *(default)* | Pause **only** on `risk` actions; `read` never prompts. |
| `step` | Confirm **every** action. |

A persistent **Kill** button (and Pause/Resume) stops the loop immediately. On a
CAPTCHA or bot check, Pilot **pauses and hands control to you** — it ships no
evasion or CAPTCHA-solving.

---

## How recipes work

1. The first successful run of a task with a `recipe` name records its concrete
   steps (URLs, durable locators, actions) to `recipes/<name>.json`.
2. Later runs **replay** those steps directly — **no model calls** — so the
   common path is fast, free and rate-limit-proof.
3. If a replayed step fails (locator missing, layout changed), Pilot re-perceives
   the live page and asks the model to re-plan, then continues.

Replay still respects approval gates: a recorded checkout step pauses in
`checkpoint` mode just like a live one.

---

## Output

Each run writes to `runs/<timestamp>/`:

- `report.md` — a clean markdown list with links **and** a cross-site comparison
  table.
- `report.json` and `report.csv` — the same items, machine-readable.
- `step_NN.png` — a screenshot per step.

---

## Adding a provider

Providers live in `pilot/providers/` and implement **one** method:

```python
from pilot.providers.base import Provider
from pilot.schemas import Action, PageState, StepRecord

class MyProvider(Provider):
    name = "mine"
    async def decide(self, goal: str, page_state: PageState,
                     history: list[StepRecord]) -> Action:
        ...
```

The **action schema is centralized** in `pilot/providers/base.py`
(`ACTION_JSON_SCHEMA` + `ACTION_GUIDE` + `build_user_text`/`parse_action`), so a
new provider only formats requests for its API — the action vocabulary is shared.
Register it in `pilot/providers/__init__.py::get_provider`. API keys are read from
the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) **at call time** and are
never stored — set them yourself as OS environment variables; nothing is written
to the repo.

### Choosing the model

You don't need to edit source. Set the model id, in precedence order:

1. **Runtime** — CLI `--model claude-sonnet-4-6`, or the UI "Model" field.
2. **Per task** — `"model": "claude-sonnet-4-6"` in the task JSON.
3. **Default** — the provider's built-in default (Anthropic: `claude-opus-4-8`).

For routine browser automation, **`claude-sonnet-4-6`** is the recommended
workhorse (fast, cheap, strong tool-use); reserve Opus for hard sites. For the
local provider, set `OLLAMA_MODEL` instead.

### Local LLM (Ollama) — no cloud, no API key

For a fully-local model, install [Ollama](https://ollama.com), pull a model, and
select the `local` provider:

```bash
ollama pull llama3.1            # or any chat/vision-capable model
python -m pilot.cli run tasks/jobs_gather.json --provider local
```

It talks to Ollama over HTTP (no extra Python dependency). Configure with
`OLLAMA_HOST` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default
`llama3.1`). Nothing leaves your machine. Note that small local models are less
reliable than the cloud models at grounding (picking the right element ref) and
at emitting valid action JSON — recipes still cover the repeatable common path
with no model at all.

---

## The five smoke tests

Run headless and offline against bundled fixtures (`tests/fixtures/`):

```bash
uv run pytest        # or: python -m pytest -q   /   make test
```

1. **DOM summary quality** — expected interactive elements, stable refs, correct
   visible/needs-scroll flags, with a canonical snapshot diff.
2. **End-to-end loop** with `StubProvider` → markdown + JSON + CSV.
3. **Recipe record & replay** — second run replays with zero provider calls and
   identical output.
4. **Approval gating** — `checkpoint` pauses only on the `risk` action.
5. **Vision fallback** — a canvas page switches perception to vision mode (logged).

> The smoke tests use a local fixture that mirrors
> [books.toscrape.com](https://books.toscrape.com) (the named demo site) so CI
> stays deterministic and network-free. Point a task at the live site to try the
> real thing.

---

## Project structure

```
pilot/
  browser.py        # Playwright wrapper + get_dom_summary
  dom_summary.js    # the single injected perception pass
  perception.py     # hybrid perception (DOM-first, vision fallback)
  providers/        # base + stub + anthropic + openai (swappable)
  agent.py          # perceive->decide->(confirm)->act loop, risk + approval, kill/pause
  recipes.py        # record once, replay deterministically
  tasks.py          # task schema + plain-code comparison/ranking
  output.py         # markdown/JSON/CSV reporters
  runner.py         # ties it all together
  server.py         # FastAPI app + web UI
  web/              # vanilla-JS UI
tasks/              # example task templates
tests/              # the five smoke tests + fixtures
recipes/            # recorded recipes (generated)
profiles/           # persistent browser profile (git-ignored)
runs/               # per-run artifacts (git-ignored)
```

---

## Legal & responsible use

**Please read this.** Pilot is a power tool that acts under your own identity.

- **You are responsible** for which sites you target and what Pilot does there.
  Automated access may violate a site's **Terms of Service** and can lead to
  rate-limiting, blocking, or **account suspension**. When in doubt, don't.
- **It acts as you.** Pilot operates inside your own logged-in sessions. Treat
  anything it does as something you did.
- **No credential handling.** You log in manually; Pilot has no code that enters,
  reads or stores passwords. Don't add any.
- **No evasion.** Pilot ships **no** anti-bot circumvention, fingerprint
  spoofing, or CAPTCHA-solving. On a CAPTCHA or bot check it **pauses and hands
  control to you**.
- **Risk actions pause by default.** Spending money, submitting, deleting and
  changing settings require your explicit approval in the default mode.
- **Be polite.** Use the optional per-action delay and reasonable step limits;
  don't hammer sites.

This project is provided for lawful, personal automation of tasks you are
entitled to perform yourself. It is not intended for scraping at scale,
circumventing access controls, or violating anyone's terms. Use it accordingly.

## License

MIT — see `pyproject.toml`.
