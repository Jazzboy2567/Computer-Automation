"""Command-line entry point.

    pilot serve                 # launch the web UI on http://127.0.0.1:8000
    pilot run <task.json>       # run a task headless/headed from the terminal
    pilot demo                  # offline end-to-end demo (no network, no API key)

`run` honors approval modes: in checkpoint/step it prompts on the terminal for
gated (risk) actions; autonomous never prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import ApprovalMode, Settings
from .schemas import Action, RiskLevel
from .tasks import Task


async def _terminal_approval(action: Action, risk: RiskLevel) -> bool:
    """Ask for confirmation on the terminal (used by checkpoint/step modes)."""
    prompt = f"\n  APPROVE [{risk.value}] {action.short()}? [y/N] "
    answer = await asyncio.to_thread(input, prompt)
    return answer.strip().lower() in ("y", "yes")


def _on_event(ev: dict) -> None:
    kind = ev.get("event")
    if kind == "decision":
        print(f"  -> step {ev['step']}: {ev['summary']} [{ev['risk']}] {ev.get('reasoning') or ''}")
    elif kind == "executed":
        print(f"     {'ok' if ev['ok'] else 'ERROR: ' + str(ev.get('error'))}")
    elif kind == "perception":
        print(f"  .. step {ev['step']}: {ev['elements']} elements [{ev['mode']}] {ev['url']}")
    elif kind == "replay_step":
        print(f"  >> replay {ev['step']}: {ev['summary']} [{ev['risk']}]")
    elif kind == "finished":
        print(f"  == {ev['message']}")


async def _run_task(args) -> int:
    from .runner import run_task

    task = Task.load(args.task)
    settings = Settings(
        headed=not args.headless,
        approval_mode=ApprovalMode(args.approval),
        provider=args.provider or task.provider or "stub",
        action_delay=args.delay,
        model=args.model,
    )
    approval = None if settings.approval_mode is ApprovalMode.AUTONOMOUS else _terminal_approval
    result, paths = await run_task(
        task, settings, on_event=_on_event, approval=approval,
        use_recipe=not args.no_recipe,
    )
    print(f"\n{'OK' if result.ok else 'FAILED'}: {result.message}")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    return 0 if result.ok else 1


async def _demo(args) -> int:
    """Run the bundled offline fixture end-to-end — proves the whole pipeline."""
    from .runner import run_task

    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "books.html"
    url = fixture.as_uri()
    task = Task(
        name="Offline demo (books fixture)",
        goal="Collect every book's title and price, cheapest first.",
        sites=[url],
        output_schema=["title", "price", "in_stock", "url"],
        sort=["price:asc"],
        provider="stub",
        script=[
            {"type": "goto", "url": url, "reasoning": "open the fixture catalog"},
            {"type": "extract", "selector": "article.product_pod",
             "fields": {"title": "h3 a", "price": "p.price_color",
                        "url": "h3 a@href", "in_stock": "p.instock.availability"},
             "store_as": "books", "reasoning": "extract products"},
            {"type": "done", "done": True, "reasoning": "done"},
        ],
    )
    settings = Settings(headed=not args.headless, provider="stub")
    result, paths = await run_task(task, settings, on_event=_on_event, record_recipe=False)
    print(f"\n{'OK' if result.ok else 'FAILED'}: extracted {len(result.extracted.get('books', []))} books")
    print(f"  report: {paths.get('markdown')}")
    return 0 if result.ok else 1


def _ml(args) -> int:
    """ML foreground: the engine produces the result; Ollama only plans."""
    from .ml.orchestrator import run_ml_goal

    def on_event(ev: dict) -> None:
        kind = ev.get("event")
        if kind == "workspace":
            print(f"  workspace: {ev['path']}")
        elif kind == "data":
            note = f" ({ev['note']})" if ev.get("note") else ""
            print(f"  data: {ev['source']} — {ev['rows']}x{ev['cols']}{note}")
        elif kind == "plan":
            print(f"  plan [{ev['planner']}]: {ev['spec']}")
        elif kind == "result":
            print(f"  result: {ev['model']} -> {ev['headline']}")

    result, ws = run_ml_goal(
        args.goal, data_path=args.data, target=args.target, planner=args.planner,
        on_event=on_event,
    )
    print(f"\nOK: {result.task_type} — {result.headline()}")
    print(f"  report: {ws.path / 'report.md'}")
    return 0


def _rl(args) -> int:
    """Train an RL game agent (toy sim, or the SPD-like trainer)."""
    def on_event(ev: dict) -> None:
        if ev.get("event") == "train":
            print(f"  training {ev['episodes']} episodes; actions={ev['actions']}")
        elif ev.get("event") == "result":
            extra = ""
            if ev.get("depth_trained") is not None:
                extra = f"; deepest floor {ev['depth_trained']} vs {ev['depth_random']}"
            print(f"  trained return {ev['trained']} vs random {ev['random']} "
                  f"(improvement {ev['improvement']:+}){extra}")

    if args.game == "spd-real":
        from .ml.rl.spd_real import bridge_available
        if not bridge_available():
            print("The headless SPD bridge is not built. Clone the SPD repo, run")
            print("`gradlew :rlbridge:writeClasspath` there, or set SPD_CLONE_DIR.")
            return 1
        if args.campaign:
            from .ml.rl.spd_campaign import run_campaign

            def on_stage(ev: dict) -> None:
                if ev.get("event") == "workspace":
                    print(f"  workspace: {ev['path']}")
                elif ev.get("event") == "stage":
                    print(f"  stage {ev['index']}: {ev['name']} "
                          f"({ev['episodes']} episodes)...", flush=True)
                elif ev.get("event") == "stage_result":
                    print(f"    return {ev['avg_return_trained']} (random {ev['avg_return_random']}), "
                          f"depth {ev['avg_depth_trained']} (max {ev['max_depth_trained']}), "
                          f"wins {ev['wins']}", flush=True)

            episodes = args.episodes if args.episodes is not None else 3000
            results, ws = run_campaign(episodes=episodes, on_event=on_stage)
            best = max(r.max_depth_trained for r in results)
            print(f"\nOK: campaign complete — {len(results)} stages, "
                  f"deepest floor reached {best}, wins {sum(r.wins for r in results)}")
            print(f"  report: {ws.path / 'report.md'}")
            return 0
        from .ml.rl.spd_campaign import challenge_mask
        from .ml.rl.spd_real_train import run_spd_real_training
        episodes = args.episodes if args.episodes is not None else 4000
        result, ws = run_spd_real_training(
            episodes=episodes, on_event=on_event,
            hero=args.hero, challenges=challenge_mask(args.challenges),
        )
        print(f"\nOK: avg return trained {result.avg_return_trained} vs random {result.avg_return_random}")
        print(f"  deepest floor: trained {result.avg_depth_trained} vs random {result.avg_depth_random}")
    elif args.game == "spd":
        from .ml.rl.spd_train import run_spd_training
        episodes = args.episodes if args.episodes is not None else 12000  # SPD needs more
        result, ws = run_spd_training(episodes=episodes, on_event=on_event)
        print(f"\nOK: avg return trained {result.avg_return_trained} vs random {result.avg_return_random}")
        print(f"  deepest floor: trained {result.avg_depth_trained} vs random {result.avg_depth_random}")
    else:
        from .ml.rl.runner import run_rl_goal
        result, ws = run_rl_goal(episodes=args.episodes or 4000, on_event=on_event)
        print(f"\nOK: avg return trained {result.avg_return_trained} vs random {result.avg_return_random}")
        print(f"  survival: trained {result.avg_survival_trained} vs random {result.avg_survival_random} steps")
    print(f"  report: {ws.path / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pilot", description="Personal browser automation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="launch the web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_run = sub.add_parser("run", help="run a task from the terminal")
    p_run.add_argument("task", help="path to a tasks/*.json file (or a bare filename)")
    p_run.add_argument("--provider", default=None, help="stub|anthropic|openai|local")
    p_run.add_argument("--model", default=None, help="model id override, e.g. claude-sonnet-4-6")
    p_run.add_argument("--approval", default="checkpoint",
                       choices=[m.value for m in ApprovalMode])
    p_run.add_argument("--headless", action="store_true")
    p_run.add_argument("--no-recipe", action="store_true", help="ignore any saved recipe")
    p_run.add_argument("--delay", type=float, default=0.0, help="seconds between actions")

    p_demo = sub.add_parser("demo", help="offline end-to-end demo (no network/API)")
    p_demo.add_argument("--headless", action="store_true", default=True)

    p_ml = sub.add_parser("ml", help="run an ML goal (ML foreground, Ollama background)")
    p_ml.add_argument("goal", help='natural-language goal, e.g. "classify iris species"')
    p_ml.add_argument("--data", default=None, help="path to a CSV (omit to use a bundled sample)")
    p_ml.add_argument("--target", default=None, help="target column to predict")
    p_ml.add_argument("--planner", default="auto", choices=["auto", "ollama", "heuristic"],
                      help="auto = Ollama if reachable, else heuristic (no AI)")

    p_rl = sub.add_parser("rl", help="train an RL game agent (screenshot->data->learn)")
    p_rl.add_argument("--game", default="sim", choices=["sim", "spd", "spd-real"],
                      help="sim = toy survival env; spd = SPD-like simulator; "
                           "spd-real = the actual game via the headless Java bridge")
    p_rl.add_argument("--episodes", type=int, default=None,
                      help="training episodes (default: 12000 for spd, 4000 for sim/spd-real)")
    p_rl.add_argument("--hero", default="warrior",
                      choices=["warrior", "mage", "rogue", "huntress", "duelist", "cleric"],
                      help="hero class (spd-real only)")
    p_rl.add_argument("--challenges", type=int, default=0, metavar="N",
                      help="enable the first N SPD challenges, 0-9 (spd-real only)")
    p_rl.add_argument("--campaign", action="store_true",
                      help="staged curriculum: long baseline, every hero class, "
                           "then challenges 1..9 (spd-real only)")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("pilot.server:app", host=args.host, port=args.port, reload=False)
        return 0
    if args.cmd == "run":
        return asyncio.run(_run_task(args))
    if args.cmd == "demo":
        return asyncio.run(_demo(args))
    if args.cmd == "ml":
        return _ml(args)
    if args.cmd == "rl":
        return _rl(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
