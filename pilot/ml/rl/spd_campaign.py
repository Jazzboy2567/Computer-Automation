"""Campaign training on the real SPD: aim for the Amulet, then generalize.

A staged curriculum over the real game, carrying the learned Q-table forward
from stage to stage:

  1. a long baseline run (warrior, no challenges) targeting depth/the Amulet,
  2. every other hero class (mage, rogue, huntress, duelist, cleric),
  3. the challenge ladder — 1 challenge enabled, then 2, ... up to all 9
     (SPD's bitmask order: NO_FOOD, NO_ARMOR, NO_HEALING, NO_HERBALISM,
     SWARM_INTELLIGENCE, DARKNESS, NO_SCROLLS, CHAMPION_ENEMIES,
     STRONGER_BOSSES).

Each stage trains with the shaped reward and is then evaluated with the pure
user reward (wins counted via the `won` flag). The campaign report is honest
about scale: finishing SPD outright is beyond a tabular agent — the ladder
measures how far each stage pushes depth, and the same harness will serve a
stronger (deep-RL) agent later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

import joblib

from ..workspace import MLWorkspace
from .agent import QLearningAgent
from .reward import RewardSpec
from .spd import spd_reward_spec, spd_training_reward
from .spd_real import SPDRealEnv
from .spd_sim import spd_featurizer
from .train import train

EventCb = Callable[[dict[str, Any]], None]
_EVAL_SEED = 990_000


def challenge_mask(count: int) -> int:
    """First `count` SPD challenges as a bitmask (0..9 -> 0..511)."""
    if not 0 <= count <= 9:
        raise ValueError("challenge count must be 0..9")
    return (1 << count) - 1


@dataclass
class Stage:
    name: str
    hero: str = "warrior"
    challenges: int = 0          # count 0..9 (converted to a mask)
    episodes: int = 3000


@dataclass
class StageResult:
    name: str
    hero: str
    challenges: int
    episodes: int
    avg_return_trained: float
    avg_return_random: float
    avg_depth_trained: float
    avg_depth_random: float
    max_depth_trained: int
    wins: int
    states_learned: int
    learning_curve: list[float] = field(default_factory=list)
    best_depth: int = 0          # deepest single run (training or eval)
    best_gear: str = ""          # the gear held on that run


def default_stages(episodes: int) -> list[Stage]:
    stages = [Stage("baseline-warrior", "warrior", 0, episodes * 2)]
    for hero in ("mage", "rogue", "huntress", "duelist", "cleric"):
        stages.append(Stage(f"hero-{hero}", hero, 0, episodes))
    for c in range(1, 10):
        stages.append(Stage(f"challenges-{c}", "warrior", c, episodes))
    return stages


def _evaluate(env: SPDRealEnv, policy, reward: RewardSpec, episodes: int):
    rets, depths, wins, max_depth = [], [], 0, 1
    for _ in range(episodes):
        obs = env.reset()
        done, total, deepest = False, 0.0, 1
        while not done:
            nxt, done, info = env.step(policy(spd_featurizer(obs)))
            total += reward.compute(obs, nxt, done, info)
            obs = nxt
            deepest = max(deepest, int(info.get("depth", 1)))
            if info.get("won"):
                wins += 1
        rets.append(total)
        depths.append(deepest)
        max_depth = max(max_depth, deepest)
    return mean(rets), mean(depths), max_depth, wins


def run_campaign(
    episodes: int = 3000,
    stages: Optional[list[Stage]] = None,
    base_dir: Optional[Path] = None,
    seed: int = 0,
    max_steps: int = 900,
    eval_episodes: int = 25,
    on_event: Optional[EventCb] = None,
) -> tuple[list[StageResult], MLWorkspace]:
    ws = MLWorkspace.create("SPD campaign (win the game / classes / challenges)", base_dir=base_dir)
    if on_event:
        on_event({"event": "workspace", "path": str(ws.path)})

    stages = stages or default_stages(episodes)
    reward = spd_reward_spec()
    train_reward = spd_training_reward()
    agent = QLearningAgent(SPDRealEnv.action_space, seed=seed)   # carried across stages
    results: list[StageResult] = []
    rng = random.Random(7)

    for i, stage in enumerate(stages):
        mask = challenge_mask(stage.challenges)
        if on_event:
            on_event({"event": "stage", "index": i, "name": stage.name,
                      "hero": stage.hero, "challenges": stage.challenges,
                      "episodes": stage.episodes})

        with SPDRealEnv(seed=seed + i * 1_000_000, max_steps=max_steps,
                        hero=stage.hero, challenges=mask) as env:
            curve = train(env, agent, train_reward, stage.episodes, featurizer=spd_featurizer)
            best_depth, best_gear = getattr(env, "best_depth", 0), getattr(env, "best_gear", "")

        with SPDRealEnv(seed=_EVAL_SEED + i * 10_000, max_steps=max_steps,
                        hero=stage.hero, challenges=mask) as env:
            rt, dt, max_dt, wins = _evaluate(env, agent.policy, reward, eval_episodes)
            if getattr(env, "best_depth", 0) > best_depth:
                best_depth, best_gear = getattr(env, "best_depth", 0), getattr(env, "best_gear", "")
        with SPDRealEnv(seed=_EVAL_SEED + i * 10_000, max_steps=max_steps,
                        hero=stage.hero, challenges=mask) as env:
            rr, dr, _, _ = _evaluate(
                env, lambda o: rng.choice(SPDRealEnv.action_space), reward, eval_episodes)

        result = StageResult(
            name=stage.name, hero=stage.hero, challenges=stage.challenges,
            episodes=stage.episodes,
            avg_return_trained=round(rt, 2), avg_return_random=round(rr, 2),
            avg_depth_trained=round(dt, 2), avg_depth_random=round(dr, 2),
            max_depth_trained=max_dt, wins=wins,
            states_learned=agent.states_learned, learning_curve=curve,
            best_depth=best_depth, best_gear=best_gear,
        )
        results.append(result)
        joblib.dump(agent.Q, ws.model_dir / f"policy_{i:02d}_{stage.name}.joblib")
        ws.write_json("campaign.json", [r.__dict__ for r in results])
        ws.write_text("report.md", _report(results, reward))
        if on_event:
            on_event({"event": "stage_result", **result.__dict__})

    joblib.dump(agent.Q, ws.model_dir / "policy.joblib")
    return results, ws


@dataclass
class TrackResult:
    hero: str
    reached: int            # -1 = still on base game; 0..9 = highest challenge COMPLETED
    attempts: dict[str, int] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)


def run_gated_campaign(
    episodes: int = 3000,
    heroes: tuple[str, ...] = SPDRealEnv.HEROES,
    max_attempts: int = 2,
    base_dir: Optional[Path] = None,
    seed: int = 0,
    max_steps: int = 900,
    eval_episodes: int = 25,
    on_event: Optional[EventCb] = None,
) -> tuple[list[TrackResult], MLWorkspace]:
    """Matthew's program: each class beats the base game, then challenge 1, then
    2, ... up to 9 — no jumps. Advancement is gated on an actual WIN (a run
    finished holding the Amulet). A stage that fails its gate is retrained up to
    `max_attempts` times; if still unbeaten the class's track halts there and the
    report says exactly where. The Q-table carries across everything.
    """
    ws = MLWorkspace.create("SPD gated campaign (win to advance)", base_dir=base_dir)
    if on_event:
        on_event({"event": "workspace", "path": str(ws.path)})

    reward = spd_reward_spec()
    train_reward = spd_training_reward()
    agent = QLearningAgent(SPDRealEnv.action_space, seed=seed)
    rng = random.Random(7)
    tracks: list[TrackResult] = []
    stage_no = 0

    for hero in heroes:
        track = TrackResult(hero=hero, reached=-1)
        for chal in range(0, 10):
            name = f"{hero}-chal{chal}"
            passed = False
            for attempt in range(1, max_attempts + 1):
                if on_event:
                    on_event({"event": "stage", "name": name, "attempt": attempt,
                              "episodes": episodes})
                mask = challenge_mask(chal)
                with SPDRealEnv(seed=seed + stage_no * 1_000_000, max_steps=max_steps,
                                hero=hero, challenges=mask) as env:
                    curve = train(env, agent, train_reward, episodes, featurizer=spd_featurizer)
                    best_depth, best_gear = getattr(env, "best_depth", 0), getattr(env, "best_gear", "")
                with SPDRealEnv(seed=_EVAL_SEED + stage_no * 10_000, max_steps=max_steps,
                                hero=hero, challenges=mask) as env:
                    rt, dt, max_dt, wins = _evaluate(env, agent.policy, reward, eval_episodes)
                    if getattr(env, "best_depth", 0) > best_depth:
                        best_depth, best_gear = getattr(env, "best_depth", 0), getattr(env, "best_gear", "")
                with SPDRealEnv(seed=_EVAL_SEED + stage_no * 10_000, max_steps=max_steps,
                                hero=hero, challenges=mask) as env:
                    rr, dr, _, _ = _evaluate(
                        env, lambda o: rng.choice(SPDRealEnv.action_space), reward, eval_episodes)

                result = StageResult(
                    name=f"{name}-a{attempt}", hero=hero, challenges=chal, episodes=episodes,
                    avg_return_trained=round(rt, 2), avg_return_random=round(rr, 2),
                    avg_depth_trained=round(dt, 2), avg_depth_random=round(dr, 2),
                    max_depth_trained=max_dt, wins=wins,
                    states_learned=agent.states_learned, learning_curve=curve,
                    best_depth=best_depth, best_gear=best_gear,
                )
                track.stages.append(result)
                track.attempts[name] = attempt
                stage_no += 1
                joblib.dump(agent.Q, ws.model_dir / "policy.joblib")
                ws.write_json("campaign.json",
                              [{**t.__dict__, "stages": [s.__dict__ for s in t.stages]}
                               for t in tracks + [track]])
                ws.write_text("report.md", _gated_report(tracks + [track]))
                if on_event:
                    on_event({"event": "stage_result", **result.__dict__})
                if wins > 0:
                    passed = True
                    break
            if passed:
                track.reached = chal
            else:
                break   # no jumps: the track halts at the first unbeaten gate
        tracks.append(track)
        if on_event:
            on_event({"event": "track_done", "hero": hero, "reached": track.reached})

    ws.write_text("report.md", _gated_report(tracks))
    return tracks, ws


def _gated_report(tracks: list[TrackResult]) -> str:
    lines = [
        "# SPD gated campaign — win to advance (per class, challenges 0→9, no jumps)", "",
        "Gate = a run finished holding the Amulet. `reached` is the highest",
        "challenge level COMPLETED (-1 = base game not yet beaten).", "",
        "| hero | reached | best depth | best return | stages run |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in tracks:
        best_depth = max((max(s.max_depth_trained, s.best_depth) for s in t.stages), default=1)
        best_ret = max((s.avg_return_trained for s in t.stages), default=0)
        lines.append(f"| {t.hero} | {t.reached} | {best_depth} | {best_ret} | {len(t.stages)} |")
    lines += ["", "## Best runs (deepest floor + gear held there)", ""]
    for t in tracks:
        best = max(t.stages, key=lambda s: s.best_depth, default=None)
        if best and best.best_depth > 0:
            lines.append(f"- **{t.hero}**: floor {best.best_depth} — {best.best_gear or '(starting kit)'}")
    lines += ["", "Per-stage detail lives in campaign.json.", ""]
    return "\n".join(lines) + "\n"


def _report(results: list[StageResult], reward: RewardSpec) -> str:
    lines = [
        "# SPD campaign — win the game, all heroes, challenges 0→9", "",
        "Policy is carried forward stage to stage (curriculum). Evaluation uses",
        "the pure user reward; `wins` counts runs finished holding the Amulet.", "",
        "| stage | hero | chals | episodes | return (rand) | depth (rand) | max depth | wins | states |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.hero} | {r.challenges} | {r.episodes} "
            f"| {r.avg_return_trained} ({r.avg_return_random}) "
            f"| {r.avg_depth_trained} ({r.avg_depth_random}) "
            f"| {r.max_depth_trained} | {r.wins} | {r.states_learned} |")
    best = max(results, key=lambda r: r.best_depth, default=None)
    if best and best.best_depth > 0:
        lines += ["", f"**Deepest run:** floor {best.best_depth} ({best.name}) — "
                      f"gear: {best.best_gear or '(starting kit)'}"]
    lines += [
        "",
        "> Honest scale note: finishing SPD outright is beyond a tabular Q-table;",
        "> this ladder measures depth progress per stage and is the harness a",
        "> stronger (deep-RL) agent will plug into.", "",
    ]
    return "\n".join(lines) + "\n"
