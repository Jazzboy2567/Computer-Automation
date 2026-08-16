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


def depth_curriculum(max_depth: int = 4, prob: float = 0.35, seed: int = 0):
    """A fraction of training episodes start on a deeper floor (2..max_depth).

    Breaks the loop where an agent that always dies on floor 2 never experiences
    the floors where gear, talents and boss tactics matter — so it can never
    learn that they matter. Only the episode's STARTING FLOOR changes; how to
    play from there is entirely the agent's to learn. Most episodes still start
    on floor 1 so the real task stays dominant, and evaluation never uses this.
    """
    rng = random.Random(seed)

    def pick(episode: int) -> int:
        return rng.randint(2, max_depth) if rng.random() < prob else 1

    return pick


def spd_env_factory(seed: int, hero: str = "warrior", challenges: int = 0,
                    max_steps: int = 300, curriculum_max_depth: int = 0):
    """Module-level (picklable) factory for the multiprocessing trainer's workers.
    Rebuilds the curriculum locally from params (a closure can't cross a spawn)."""
    from .spd_real import SPDRealEnv
    cur = depth_curriculum(curriculum_max_depth, seed=seed) if curriculum_max_depth else None
    return SPDRealEnv(seed=seed, hero=hero, challenges=challenges,
                      max_steps=max_steps, curriculum=cur)


def make_agent(kind: str, actions: list[str], seed: int = 0):
    """'table' = tabular Q over the compact featurizer; 'dqn' = neural net over
    the FULL observation (featurizer identity). Returns (agent, featurizer)."""
    if kind == "dqn":
        import os

        from .dqn import DQNAgent
        # hidden=128 over the ~104-feature focused encoding was the proven-fast CPU
        # config (a 256-wide trunk was much slower per step on CPU). But a batch-64
        # gradient step through this tiny net is TOO SMALL for the GPU to help
        # (measured 0.5x — kernel-launch/transfer overhead > the matmul). The GPU
        # only pays off with a BIGGER net: hidden=512 is ~36x slower than hidden=128
        # on CPU but the SAME wall-clock on the GPU (17x speedup at batch 64) — i.e.
        # the GPU unlocks a higher-capacity net for free, which the hard long-horizon
        # gear/exploration problem plausibly needs. So backend + width are env-tunable:
        #   PILOT_DQN_BACKEND=torch  PILOT_DQN_HIDDEN=512   (run under .venv-gpu)
        # Defaults keep the dependency-free CPU path (numpy, hidden 128) unchanged.
        backend = os.environ.get("PILOT_DQN_BACKEND", "numpy")
        hidden = int(os.environ.get("PILOT_DQN_HIDDEN", "128"))
        return DQNAgent(actions, seed=seed, hidden=hidden, backend=backend), spd_map_featurizer
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
    curriculum: Any = None,       # callable(episode) -> starting floor, training only
    resume_from: Optional[Path] = None,   # a policy.joblib to continue learning from
    epsilon_start: float = 1.0,           # lower this when resuming (see train())
    on_event: Optional[EventCb] = None,
) -> tuple[RLResult, MLWorkspace]:
    ws = MLWorkspace.create("Shattered Pixel Dungeon (REAL game)", base_dir=base_dir)
    _emit(on_event, event="workspace", path=str(ws.path))

    reward = spd_reward_spec()             # the user's true objective (for eval)
    train_reward = spd_training_reward()   # + shaping toward the stairs (for learning)
    agent, feat = make_agent(agent_kind, SPDRealEnv.action_space, seed)
    if resume_from is not None and Path(resume_from).exists():
        # continue learning from an earlier chunk instead of starting over, so
        # training can run indefinitely across many runs
        agent.Q = joblib.load(resume_from)
        _emit(on_event, event="resume", path=str(resume_from))
    _emit(on_event, event="train", episodes=episodes, actions=SPDRealEnv.action_space)

    kw = {"max_steps": max_steps, "hero": hero, "challenges": challenges}
    best_depth, best_gear = 0, ""
    # Curriculum applies to TRAINING only; every evaluation below starts on floor
    # 1 so the reported numbers always measure the real task.
    with SPDRealEnv(seed=seed, curriculum=curriculum, **kw) as env:
        curve = train(env, agent, train_reward, episodes, featurizer=feat,
                      epsilon_start=epsilon_start)
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


def run_continuous(
    *,
    interval: int = 4000,
    base_dir: Optional[Path] = None,
    seed: int = 0,
    max_steps: int = 600,
    eval_episodes: int = 30,
    hero: str = "warrior",
    challenges: int = 0,
    agent_kind: str = "dqn",
    curriculum: Any = None,
    decay_episodes: int = 40000,
    on_interval: Optional[Callable[[dict], None]] = None,
):
    """Train ONE agent indefinitely, reporting every `interval` episodes.

    Unlike calling run_spd_real_training in a loop, the agent — and therefore its
    replay buffer and optimizer state — lives for the whole run, so experience
    genuinely ACCUMULATES (the rare deep-floor transitions build up instead of
    being thrown away each chunk) and exploration decays ONCE, globally, from 1.0
    to 0.05 over `decay_episodes`, rather than restarting every chunk.

    Evaluation replays the SAME floor-1 dungeons each interval (eval env seed reset),
    so the reported curve reflects real change, not eval-set noise. Runs until the
    caller stops the process; every interval leaves a saved policy behind.
    """
    ws = MLWorkspace.create("Shattered Pixel Dungeon (REAL game, continuous)", base_dir=base_dir)
    reward = spd_reward_spec()
    train_reward = spd_training_reward()
    agent, feat = make_agent(agent_kind, SPDRealEnv.action_space, seed)
    kw = {"max_steps": max_steps, "hero": hero, "challenges": challenges}

    def eps_at(e: int) -> float:
        return max(0.05, 1.0 - (1.0 - 0.05) * e / max(1, decay_episodes))

    train_env = SPDRealEnv(seed=seed, curriculum=curriculum, **kw)
    eval_env = SPDRealEnv(seed=_EVAL_SEED, **kw)
    # random baseline is constant — measure it once
    rng = random.Random(7)
    rr, sr, dr = _evaluate(eval_env, lambda o: rng.choice(SPDRealEnv.action_space),
                           reward, eval_episodes)
    eval_env.best_depth, eval_env.best_gear = 0, ""   # don't credit random's floors

    done_eps, k = 0, 0
    best_score = None                                 # (avg_depth, avg_return) high-water mark
    env_restarts, consecutive_crashes = 0, 0
    try:
        while True:
            k += 1
            e0, e1 = eps_at(done_eps), eps_at(done_eps + interval)
            try:
                curve = train(train_env, agent, train_reward, interval, featurizer=feat,
                              epsilon_start=e0, epsilon_final=e1)
                eval_env.episode = 0                  # replay the same eval dungeons
                rt, st, dt = _evaluate(eval_env, agent.policy, reward, eval_episodes, feat=feat)
            except RuntimeError as e:
                # The headless bridge is a JVM and can die mid-episode — a native/OOM
                # crash (no Python-catchable stack) or un-soaked deep content. For an
                # unattended run that must NEVER be true: rebuild the env(s) and carry
                # on. The agent — replay buffer, weights, optimizer — outlives the env,
                # so only the in-flight interval is lost, not the run. A fresh training
                # seed each restart avoids re-triggering a seed-specific crash; if it
                # crashes 8 intervals running, something is truly broken, so abort
                # rather than spin. This is what lets a deep-diving agent (which reaches
                # the least-soaked content, exactly where crashes hide) keep training.
                env_restarts += 1
                consecutive_crashes += 1
                print(f"[interval {k}] EnvServer died ({e!r}); restarting envs "
                      f"(restart #{env_restarts}) and continuing", flush=True)
                for ev in (train_env, eval_env):
                    try:
                        ev.close()
                    except Exception:
                        pass
                if consecutive_crashes >= 8:
                    print("[continuous] 8 consecutive env crashes — aborting to avoid a spin loop", flush=True)
                    break
                train_env = SPDRealEnv(seed=seed + 100_000 * env_restarts, curriculum=curriculum, **kw)
                eval_env = SPDRealEnv(seed=_EVAL_SEED, **kw)
                eval_env.best_depth, eval_env.best_gear = 0, ""
                k -= 1                                # this interval didn't complete
                continue
            consecutive_crashes = 0
            done_eps += interval

            joblib.dump(agent.Q, ws.model_dir / "policy.joblib")   # latest, always
            # keep the best-so-far separately, so a long run that later drifts or
            # destabilises never loses its high point (depth first, then return)
            new_best = ""
            if best_score is None or (dt, rt) > best_score:
                best_score = (dt, rt)
                joblib.dump(agent.Q, ws.model_dir / "policy_best.joblib")
                new_best = "  <== new best"

            info = {
                "interval": k, "total_eps": done_eps, "eps": round(e1, 3),
                "return": round(rt, 2), "return_random": round(rr, 2),
                "depth": round(dt, 2), "survival": round(st, 1),
                "best_depth": eval_env.best_depth, "best_gear": eval_env.best_gear,
                "curve_start": curve[0], "curve_end": curve[-1],
                "new_best": new_best, "workspace": str(ws.path),
            }
            if on_interval:
                # a reporting error must never throw away hours of training
                try:
                    on_interval(info)
                except Exception as e:
                    print(f"[interval {k}] report failed (continuing): {e!r}", flush=True)
    finally:
        for ev in (train_env, eval_env):
            try:
                ev.close()
            except Exception:
                pass
