"""Run an RL goal end-to-end in its own workspace (simulated game for now).

Trains the agent, evaluates it against a random baseline, saves the learned
policy, and writes a report — mirroring the rest of the ML foreground.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Optional

import joblib

from ..workspace import MLWorkspace
from .agent import QLearningAgent
from .env import SimEnv
from .reward import RewardSpec
from .train import RLResult, evaluate, train

EventCb = Callable[[dict[str, Any]], None]
_EVAL_SEED = 999


def _emit(cb: Optional[EventCb], **event: Any) -> None:
    if cb:
        cb(event)


def _transcript(agent: QLearningAgent, reward: RewardSpec) -> str:
    """One greedy episode, rendered, so you can see what the agent does."""
    env = SimEnv(seed=1)
    obs = env.reset()
    lines = [f"start: hp={obs['player_health']:.0f} enemies={obs['enemies_nearby']:.0f}"]
    done, total = False, 0.0
    while not done:
        a = agent.policy(obs)
        nxt, done, info = env.step(a)
        r = reward.compute(obs, nxt, done, info)
        total += r
        lines.append(f"{a:<6} -> hp={nxt['player_health']:.0f} enemies={nxt['enemies_nearby']:.0f}  r={r:+.1f}")
        obs = nxt
    lines.append(f"return = {total:.1f}")
    return "\n".join(lines)


def _report(goal: str, result: RLResult, reward: RewardSpec) -> str:
    lines = [
        f"# RL goal: {goal}", "",
        f"**Result:** {result.headline()}  ",
        f"**Improvement over random:** {result.improvement:+.1f}  ",
        f"**Episodes:** {result.episodes} · **states learned:** {result.states_learned}",
        "",
        "## Performance (trained vs random)", "",
        "| metric | trained | random |", "| --- | --- | --- |",
        f"| avg return | {result.avg_return_trained} | {result.avg_return_random} |",
        f"| avg survival (steps) | {result.avg_survival_trained} | {result.avg_survival_random} |",
        "",
        "## Reward spec (your good/bad events)", "",
        "```json", reward.model_dump_json(indent=2), "```", "",
        "## Learning curve (mean return per block)", "",
        "`" + " ".join(str(x) for x in result.learning_curve) + "`", "",
        "## Sample episode (trained, greedy)", "",
        "```", result.policy_sample or "", "```", "",
        "## Artifacts", "",
        f"- policy: `{result.model_path}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_rl_goal(
    goal: str = "learn to survive (simulated game)",
    episodes: int = 4000,
    reward: Optional[RewardSpec] = None,
    base_dir: Optional[Path] = None,
    seed: int = 0,
    on_event: Optional[EventCb] = None,
) -> tuple[RLResult, MLWorkspace]:
    """Train + evaluate an RL agent on the simulated game; write a workspace report."""
    ws = MLWorkspace.create(goal, base_dir=base_dir)
    _emit(on_event, event="workspace", path=str(ws.path))

    reward = reward or RewardSpec.survival_default()
    env = SimEnv(seed=seed)
    agent = QLearningAgent(env.action_space, bins=env.discretization(), seed=seed)
    _emit(on_event, event="train", episodes=episodes, actions=env.action_space)

    curve = train(env, agent, reward, episodes)

    # Fair comparison: trained and random face identically-seeded episodes.
    ret_t, surv_t = evaluate(SimEnv(seed=_EVAL_SEED), agent.policy, reward, 200)
    rng = random.Random(7)
    ret_r, surv_r = evaluate(SimEnv(seed=_EVAL_SEED), lambda o: rng.choice(env.action_space), reward, 200)

    model_path = ws.model_dir / "policy.joblib"
    joblib.dump(agent.Q, model_path)

    result = RLResult(
        episodes=episodes, actions=env.action_space,
        avg_return_trained=round(ret_t, 2), avg_return_random=round(ret_r, 2),
        avg_survival_trained=round(surv_t, 1), avg_survival_random=round(surv_r, 1),
        improvement=round(ret_t - ret_r, 2), states_learned=agent.states_learned,
        learning_curve=curve, model_path=str(model_path),
        policy_sample=_transcript(agent, reward),
    )
    ws.write_json("metrics.json", result.model_dump())
    ws.write_json("reward_spec.json", reward.model_dump())
    ws.write_text("report.md", _report(goal, result, reward))
    _emit(on_event, event="result", trained=result.avg_return_trained,
          random=result.avg_return_random, improvement=result.improvement)
    _emit(on_event, event="report", path=str(ws.path / "report.md"))
    return result, ws
