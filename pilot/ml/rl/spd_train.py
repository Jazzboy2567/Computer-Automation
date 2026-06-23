"""Train the agent on the SPD-like simulator and report how well it performs.

Train-first, deploy-by-mouse-later: this produces a learned policy (Q-table) over
SPD's decision-state; later it drives the real game through the
screenshot -> extract -> mouse-click seam.
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
from .spd_sim import SPDGridEnv, spd_featurizer
from .train import RLResult, train

EventCb = Callable[[dict[str, Any]], None]
_EVAL_SEED = 4242


def _emit(cb: Optional[EventCb], **event: Any) -> None:
    if cb:
        cb(event)


def _evaluate(env: SPDGridEnv, policy, reward: RewardSpec, episodes: int) -> tuple[float, float, float]:
    """Return (mean return, mean survived steps, mean deepest floor)."""
    rets, survs, depths = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done, total, steps, deepest = False, 0.0, 0, 1
        while not done:
            action = policy(spd_featurizer(obs))
            nxt, done, info = env.step(action)
            total += reward.compute(obs, nxt, done, info)
            obs = nxt
            steps += 1
            deepest = max(deepest, int(info.get("depth", 1)))
        rets.append(total)
        survs.append(steps)
        depths.append(deepest)
    return mean(rets), mean(survs), mean(depths)


def _transcript(agent: QLearningAgent, reward: RewardSpec) -> str:
    env = SPDGridEnv(seed=1)
    obs = env.reset()
    lines, done, total = [], False, 0.0
    while not done and len(lines) < 40:
        a = agent.policy(spd_featurizer(obs))
        nxt, done, info = env.step(a)
        total += reward.compute(obs, nxt, done, info)
        lines.append(f"{a:<8} -> hp={nxt['hp_current']:.0f}/{nxt['hp_max']:.0f} "
                     f"depth={nxt['depth']:.0f} lvl={nxt['level']:.0f} "
                     f"enemies={nxt['enemies_visible']:.0f}")
        obs = nxt
    lines.append(f"return = {total:.1f}")
    return "\n".join(lines)


def _report(result: RLResult, reward: RewardSpec) -> str:
    return "\n".join([
        "# Shattered Pixel Dungeon — agent (sim training)", "",
        f"**Result:** {result.headline()}  ",
        f"**Improvement over random:** {result.improvement:+.1f}  ",
        f"**Episodes:** {result.episodes} · **states learned:** {result.states_learned}",
        "",
        "> Trained on an SPD-like simulator. Deploy onto the real game via the "
        "screenshot → feature extractor → mouse-click seam; expect to fine-tune "
        "(sim-to-real gap).",
        "",
        "## Performance (trained vs random)", "",
        "| metric | trained | random |", "| --- | --- | --- |",
        f"| avg return | {result.avg_return_trained} | {result.avg_return_random} |",
        f"| avg survival (steps) | {result.avg_survival_trained} | {result.avg_survival_random} |",
        f"| avg deepest floor | {result.avg_depth_trained} | {result.avg_depth_random} |",
        "",
        "## Reward spec (your good/bad events)", "",
        "```json", reward.model_dump_json(indent=2), "```", "",
        "## Learning curve (mean return per block)", "",
        "`" + " ".join(str(x) for x in result.learning_curve) + "`", "",
        "## Sample episode (trained, greedy)", "",
        "```", result.policy_sample or "", "```", "",
        "## Artifacts", "",
        f"- policy: `{result.model_path}`", "",
    ]) + "\n"


def run_spd_training(
    episodes: int = 8000,
    base_dir: Optional[Path] = None,
    seed: int = 0,
    on_event: Optional[EventCb] = None,
) -> tuple[RLResult, MLWorkspace]:
    ws = MLWorkspace.create("Shattered Pixel Dungeon (sim training)", base_dir=base_dir)
    _emit(on_event, event="workspace", path=str(ws.path))

    reward = spd_reward_spec()                 # the user's true objective (for eval)
    train_reward = spd_training_reward()       # + shaping toward the stairs (for learning)
    env = SPDGridEnv(seed=seed)
    agent = QLearningAgent(SPDGridEnv.action_space, seed=seed)
    _emit(on_event, event="train", episodes=episodes, actions=SPDGridEnv.action_space)

    curve = train(env, agent, train_reward, episodes, featurizer=spd_featurizer)

    rt, st, dt = _evaluate(SPDGridEnv(seed=_EVAL_SEED), agent.policy, reward, 200)
    rng = random.Random(7)
    rr, sr, dr = _evaluate(SPDGridEnv(seed=_EVAL_SEED),
                           lambda o: rng.choice(SPDGridEnv.action_space), reward, 200)

    model_path = ws.model_dir / "policy.joblib"
    joblib.dump(agent.Q, model_path)

    result = RLResult(
        episodes=episodes, actions=SPDGridEnv.action_space,
        avg_return_trained=round(rt, 2), avg_return_random=round(rr, 2),
        avg_survival_trained=round(st, 1), avg_survival_random=round(sr, 1),
        improvement=round(rt - rr, 2), states_learned=agent.states_learned,
        learning_curve=curve, model_path=str(model_path),
        avg_depth_trained=round(dt, 2), avg_depth_random=round(dr, 2),
        policy_sample=_transcript(agent, reward),
    )
    ws.write_json("metrics.json", result.model_dump())
    ws.write_json("reward_spec.json", reward.model_dump())
    ws.write_text("report.md", _report(result, reward))
    _emit(on_event, event="result", trained=result.avg_return_trained,
          random=result.avg_return_random, improvement=result.improvement,
          depth_trained=result.avg_depth_trained, depth_random=result.avg_depth_random)
    return result, ws
