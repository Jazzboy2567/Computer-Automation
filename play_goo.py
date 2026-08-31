"""Play the Goo fight AS THE AGENT — same headless game, same observation, same 32
actions — to record EXPERT DEMONSTRATIONS. Your winning runs are saved to
goo_demos.joblib and can be seeded into the agent's replay buffer so it learns YOUR
strategy (imitation from a human expert, not a scripted tactic).

No GUI: you see exactly what the agent sees (your HP, Goo's HP %, whether Goo is
pumping up, your gear, adjacent enemies) and you pick one action each turn.

Run it in your own terminal (it needs keyboard input):
    .venv\\Scripts\\python.exe play_goo.py
Optional: `--floor 1` to play a full run from floor 1 instead of starting at Goo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PILOT_SEED_ACQ_KIT_PROB", "1.0")  # gear-up materials available

import joblib

from pilot.ml.rl.spd import SPD_ACTIONS, spd_training_reward
from pilot.ml.rl.spd_real import SPDRealEnv

DEMO_FILE = Path(__file__).with_name("goo_demos.joblib")
REWARD = spd_training_reward()


def _fmt_actions() -> str:
    lines, row = [], []
    for i, a in enumerate(SPD_ACTIONS):
        row.append(f"{i:2d} {a}")
        if len(row) == 4:
            lines.append("   ".join(row)); row = []
    if row:
        lines.append("   ".join(row))
    return "\n".join("  " + ln for ln in lines)


def _show(obs: dict, info: dict, turn: int) -> None:
    depth = int(obs.get("depth", 5))
    gear = info.get("gear", "") or "(starting kit)"
    print(f"\n=== Turn {turn} | Floor {depth} ===")
    print(f"  You: HP {obs.get('hp_current', 0):.0f}/{obs.get('hp_max', 0):.0f}"
          f"   STR {obs.get('str', 0):.0f}   gear: {gear}")
    boss = float(obs.get("boss_hp_frac", 0) or 0)
    charge = float(obs.get("enemy0_charging", 0) or 0)
    dx, dy = int(obs.get("enemy0_dx", 0)), int(obs.get("enemy0_dy", 0))
    if boss > 0:
        adj = "(ADJACENT)" if max(abs(dx), abs(dy)) <= 1 else ""
        tag = "   <<< GOO IS PUMPING UP! >>>" if charge >= 1 else ""
        print(f"  GOO: {boss * 100:3.0f}% HP   at dx={dx:+d} dy={dy:+d} {adj}{tag}")
    elif float(obs.get("enemy0_hp", 0) or 0) > 0:
        print(f"  Enemy: {float(obs.get('enemy0_hp', 0)) * 100:3.0f}% HP"
              f"   at dx={dx:+d} dy={dy:+d}   charging={charge:.0f}")
    else:
        hint = "  — walk in to find Goo (try 'explore', or move toward the arena)" if depth >= 5 else ""
        print(f"  (no enemy in view){hint}")
    print(f"  Bag: str_potions={obs.get('str_potions', 0):.0f}"
          f"  upgrade_scrolls={obs.get('upgrade_scrolls', 0):.0f}"
          f"  heal_potion={'yes' if obs.get('has_heal', 0) else 'no'}"
          f"  gear_to_equip={'yes' if obs.get('gear_available', 0) else 'no'}")
    print(f"  Adjacent enemies: {obs.get('enemies_adjacent', 0):.0f}"
          f"   |   stairs {'seen' if obs.get('stairs_dist', 30) < 30 else 'not found'}")


def _prompt() -> str:
    while True:
        raw = input("  Move (# / name, or 'menu' 'restart' 'quit'): ").strip().lower()
        if raw in ("quit", "q"):
            return "__quit__"
        if raw in ("restart", "r"):
            return "__restart__"
        if raw in ("menu", "m", "?", "help"):
            print(_fmt_actions()); continue
        if raw.isdigit() and int(raw) < len(SPD_ACTIONS):
            return SPD_ACTIONS[int(raw)]
        if raw in SPD_ACTIONS:
            return raw
        print("   ? unknown action — type 'menu' to list them.")


def _load_demos() -> list:
    if DEMO_FILE.exists():
        try:
            return joblib.load(DEMO_FILE)
        except Exception:
            pass
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=5,
                    help="starting floor (5 = straight into the Goo fight with gear; 1 = full run)")
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args()

    demos = _load_demos()
    print(__doc__)
    print(f"Loaded {len(demos)} demo transitions so far (file: {DEMO_FILE.name}).")
    print("\nActions you can take:")
    print(_fmt_actions())
    print("\nGoal: kill Goo and take the stairs down (reach floor 6). Good luck.\n")

    curriculum = (lambda ep: args.floor) if args.floor > 1 else None
    env = SPDRealEnv(seed=args.seed, max_steps=300, curriculum=curriculum)
    wins = 0
    try:
        while True:
            obs = env.reset()
            info = {"gear": ""}
            traj, turn, done = [], 0, False
            reached = int(obs.get("depth", args.floor))
            quit_all = False
            while not done:
                turn += 1
                _show(obs, info, turn)
                action = _prompt()
                if action == "__quit__":
                    quit_all = True; break
                if action == "__restart__":
                    print("  -- restarting episode (not saved) --"); break
                prev = obs
                obs, done, info = env.step(action)
                reward = REWARD.compute(prev, obs, done, info)
                traj.append({"obs": prev, "action": action, "reward": float(reward),
                             "next_obs": obs, "done": bool(done)})
                reached = max(reached, int(obs.get("depth", args.floor)))
                if float(prev.get("boss_hp_frac", 0) or 0) > 0 and float(obs.get("boss_hp_frac", 0) or 0) == 0:
                    print("  *** Goo down! ***")
            if quit_all:
                break
            if done:
                won = reached >= 6
                print(f"\n  Episode over — {'WON (reached floor 6, Goo dead)!' if won else 'lost.'}"
                      f"  (deepest floor {reached}, {turn} turns)")
                if won:
                    demos.extend(traj)
                    wins += 1
                    joblib.dump(demos, DEMO_FILE)
                    print(f"  Saved this win: {len(traj)} transitions "
                          f"(total demo transitions now {len(demos)} across your wins).")
                else:
                    keep = input("  Save this run anyway as a demo? (y/N): ").strip().lower()
                    if keep == "y":
                        demos.extend(traj); joblib.dump(demos, DEMO_FILE)
                        print(f"  Saved ({len(demos)} total).")
            again = input("\n  Play another? (Y/n): ").strip().lower()
            if again == "n":
                break
    finally:
        env.close()
    print(f"\nDone. {wins} win(s) this session; {len(demos)} demo transitions in {DEMO_FILE.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
