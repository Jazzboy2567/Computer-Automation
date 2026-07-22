"""Train the agent on the REAL Shattered Pixel Dungeon (headless Java bridge).

Same trainer, featurizer, and reward as the sim pipeline — but the environment
is the actual game, so there is no sim-to-real gap to close afterwards.
"""

from __future__ import annotations

import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

import joblib

from ..workspace import MLWorkspace
from .agent import QLearningAgent
from .reward import RewardSpec
from .spd import spd_reward_spec, spd_training_reward
from .spd_real import SPDRealEnv
from .spd_sim import spd_featurizer, spd_map_featurizer
from .train import RLResult, train

EventCb = Callable[[dict[str, Any]], None]
_EVAL_SEED = 990_000  # evaluation dungeons never overlap training seeds


def _emit(cb: Optional[EventCb], **event: Any) -> None:
    if cb:
        cb(event)


def _evaluate(env: SPDRealEnv, policy, reward: RewardSpec, episodes: int,
              feat=spd_featurizer) -> tuple[float, float, float]:
    """Return (mean return, mean survived steps, mean deepest floor)."""
    rets, survs, depths = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done, total, steps, deepest = False, 0.0, 0, 1
        while not done:
            action = policy(feat(obs))
            nxt, done, info = env.step(action)
            total += reward.compute(obs, nxt, done, info)
            obs = nxt
            steps += 1
            deepest = max(deepest, int(info.get("depth", 1)))
        rets.append(total)
        survs.append(steps)
        depths.append(deepest)
    return mean(rets), mean(survs), mean(depths)


def _report(result: RLResult, reward: RewardSpec) -> str:
    return "\n".join([
        "# Shattered Pixel Dungeon — agent (REAL game, headless)", "",
        f"**Result:** {result.headline()}  ",
        f"**Improvement over random:** {result.improvement:+.1f}  ",
        f"**Episodes:** {result.episodes} · **states learned:** {result.states_learned}",
        "",
        "> Trained on the actual open-source SPD running headless — the dynamics",
        "> are the real game's, so there is no sim-to-real gap. Observations are",
        "> strictly player-visible (fog of war respected).",
        "",
        "## Performance (trained vs random)", "",
        "| metric | trained | random |", "| --- | --- | --- |",
        f"| avg return | {result.avg_return_trained} | {result.avg_return_random} |",
        f"| avg survival (actions) | {result.avg_survival_trained} | {result.avg_survival_random} |",
        f"| avg deepest floor | {result.avg_depth_trained} | {result.avg_depth_random} |",
        "",
        f"**Best run:** floor {result.best_depth} — gear: {result.best_gear or '(starting kit)'}",
        "",
        "## Reward spec (your good/bad events)", "",
        "```json", reward.model_dump_json(indent=2), "```", "",
        "## Learning curve (mean return per block)", "",
        "`" + " ".join(str(x) for x in result.learning_curve) + "`", "",
        "## Artifacts", "",
        f"- policy: `{result.model_path}`", "",
    ]) + "\n"


def make_agent(kind: str, actions: list[str], seed: int = 0):
    """'table' = tabular Q over the compact featurizer; 'dqn' = neural net over
    the FULL observation (featurizer identity). Returns (agent, featurizer)."""
    if kind == "dqn":
        from .dqn import DQNAgent
        # bigger hidden layer: the input now includes the ~810-cell egocentric map
        return DQNAgent(actions, seed=seed, hidden=128), spd_map_featurizer
    return QLearningAgent(actions, seed=seed), spd_featurizer


def run_spd_real_training(
    episodes: int = 4000,
    base_dir: Optional[Path] = None,
    seed: int = 0,
    max_steps: int = 600,
    eval_episodes: int = 30,
    hero: str = "warrior",
    challenges: int = 0,          # SPD challenge bitmask
    agent_kind: str = "table",
    on_event: Optional[EventCb] = None,
) -> tuple[RLResult, MLWorkspace]:
    ws = MLWorkspace.create("Shattered Pixel Dungeon (REAL game)", base_dir=base_dir)
    _emit(on_event, event="workspace", path=str(ws.path))

    reward = spd_reward_spec()             # the user's true objective (for eval)
    train_reward = spd_training_reward()   # + shaping toward the stairs (for learning)
    agent, feat = make_agent(agent_kind, SPDRealEnv.action_space, seed)
    _emit(on_event, event="train", episodes=episodes, actions=SPDRealEnv.action_space)

    kw = {"max_steps": max_steps, "hero": hero, "challenges": challenges}
    best_depth, best_gear = 0, ""
    with SPDRealEnv(seed=seed, **kw) as env:
        curve = train(env, agent, train_reward, episodes, featurizer=feat)
        best_depth, best_gear = getattr(env, "best_depth", 0), getattr(env, "best_gear", "")

    with SPDRealEnv(seed=_EVAL_SEED, **kw) as env:
        rt, st, dt = _evaluate(env, agent.policy, reward, eval_episodes, feat=feat)
        if getattr(env, "best_depth", 0) > best_depth:
            best_depth, best_gear = getattr(env, "best_depth", 0), getattr(env, "best_gear", "")
    rng = random.Random(7)
    with SPDRealEnv(seed=_EVAL_SEED, **kw) as env:
        rr, sr, dr = _evaluate(env, lambda o: rng.choice(SPDRealEnv.action_space),
                               reward, eval_episodes)

    model_path = ws.model_dir / "policy.joblib"
    joblib.dump(agent.Q, model_path)

    result = RLResult(
        episodes=episodes, actions=SPDRealEnv.action_space,
        avg_return_trained=round(rt, 2), avg_return_random=round(rr, 2),
        avg_survival_trained=round(st, 1), avg_survival_random=round(sr, 1),
        improvement=round(rt - rr, 2), states_learned=agent.states_learned,
        learning_curve=curve, model_path=str(model_path),
        avg_depth_trained=round(dt, 2), avg_depth_random=round(dr, 2),
        best_depth=best_depth, best_gear=best_gear,
    )
    ws.write_json("metrics.json", result.model_dump())
    ws.write_json("reward_spec.json", reward.model_dump())
    ws.write_text("report.md", _report(result, reward))
    _emit(on_event, event="result", trained=result.avg_return_trained,
          random=result.avg_return_random, improvement=result.improvement,
          depth_trained=result.avg_depth_trained, depth_random=result.avg_depth_random)
    return result, ws
