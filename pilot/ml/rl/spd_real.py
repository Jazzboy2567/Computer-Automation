"""Train on the *real* Shattered Pixel Dungeon via the headless Java bridge.

The open-source SPD (Java) runs headless in an external clone with a small
``rlbridge`` module: an ``EnvServer`` speaking a line protocol over
stdin/stdout — ``reset <seed>`` / ``act <action>`` / ``quit`` in, one JSON
observation per line out. Observations are strictly player-visible (enemies
through the hero's field of view; stairs only once the exit tile has been
seen), so the agent never learns from information a player wouldn't have.

This module wraps that process as a `GameEnv`, so the existing tabular
trainer, featurizer, and reward spec run against the real game unchanged.
Zero sim-to-real gap: the dynamics ARE the game.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .env import GameEnv, Observation
from .spd import SPD_ACTIONS

# Where the SPD clone (with the rlbridge module) lives; override via env var.
SPD_CLONE_ENV = "SPD_CLONE_DIR"
DEFAULT_CLONE = Path.home() / "shattered-pixel-dungeon"

# stairs_dist while the exit is undiscovered (server reports -1). A constant
# "assume far" keeps the potential-based shaping artifact-free: discovering the
# stairs then reads as progress (dist drops), never as a spurious penalty.
UNKNOWN_STAIRS_DIST = 30.0

# Reported by the server but not part of the agent's observation.
_INFO_FIELDS = ("done", "turns", "pos", "gear")


def clone_dir() -> Path:
    return Path(os.environ.get(SPD_CLONE_ENV, str(DEFAULT_CLONE)))


def bridge_available(clone: Optional[Path] = None) -> bool:
    """True if the SPD clone has a compiled rlbridge classpath to launch."""
    clone = clone or clone_dir()
    return (clone / "rlbridge" / "build" / "rlbridge.classpath").exists()


def stderr_log_path() -> Path:
    """Where the Java server's stderr goes — stack traces and the headless
    safety net's recovered-exception reports end up here, not in a black hole."""
    return Path(tempfile.gettempdir()) / f"spd_envserver_{os.getpid()}.log"


def launch_server(clone: Optional[Path] = None) -> subprocess.Popen:
    """Start the Java EnvServer (compile first: `gradlew :rlbridge:writeClasspath`)."""
    clone = clone or clone_dir()
    cp_file = clone / "rlbridge" / "build" / "rlbridge.classpath"
    if not cp_file.exists():
        raise FileNotFoundError(
            f"rlbridge classpath not found at {cp_file}. Clone the SPD repo and run "
            f"`gradlew :rlbridge:writeClasspath` there, or set {SPD_CLONE_ENV}."
        )
    return subprocess.Popen(
        ["java", "-cp", cp_file.read_text(encoding="utf-8").strip(), "rlbridge.EnvServer"],
        cwd=clone / "core" / "src" / "main" / "assets",   # Gdx.files.internal resolves here
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(stderr_log_path(), "a", encoding="utf-8"),   # appended across restarts
        text=True,
        encoding="utf-8",
        errors="replace",   # a dying JVM must surface as a protocol error, not a decode crash
    )


class SPDRealEnv(GameEnv):
    """The real game as a `GameEnv`: each instance owns one Java process.

    Each ``reset()`` starts a fresh run on a new seed (``base_seed`` + episode
    index), so training sees many dungeons rather than memorizing one layout.
    Episodes end on death or after ``max_steps`` agent actions.
    """

    action_space = list(SPD_ACTIONS)

    HEROES = ("warrior", "mage", "rogue", "huntress", "duelist", "cleric")

    def __init__(self, seed: int = 0, max_steps: int = 600, proc: Any = None,
                 hero: str = "warrior", challenges: int = 0, curriculum: Any = None):
        if hero not in self.HEROES:
            raise ValueError(f"unknown hero {hero!r}; one of {self.HEROES}")
        self.base_seed = seed
        self.max_steps = max_steps
        self.hero = hero
        self.challenges = challenges          # SPD challenge bitmask
        # Optional callable (episode_index) -> starting floor. An agent that always
        # dies on floor 2 never sees the depths where gear/talents/bosses matter,
        # so it can't learn they matter; starting some episodes deeper breaks that
        # loop. Only the starting position changes — play is never scripted.
        # Evaluation leaves this None so it always measures the real task (floor 1).
        self.curriculum = curriculum
        self.start_depth = 1                  # what the last reset() actually used
        self.steps = 0
        self.episode = 0
        # best run served by this env instance (training or eval), for reporting
        self.best_depth = 0
        self.best_gear = ""
        self._proc = proc or launch_server()

    def observation_fields(self) -> list[str]:
        return ["hp_current", "hp_max", "hp_frac", "level", "xp_frac", "depth",
                "gold", "enemies_visible", "inventory_count", "starving", "has_ankh"]

    # ------------------------------------------------------------- protocol
    def _send(self, command: str) -> dict:
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"SPD EnvServer exited unexpectedly (see {stderr_log_path()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(
                f"SPD EnvServer error: {reply['error']} (see {stderr_log_path()})"
            )
        return reply

    @staticmethod
    def _to_obs(reply: dict) -> Observation:
        # scalars are floats; list fields (the egocentric map window) pass through
        # untouched for the featurizer to unpack
        obs = {}
        for k, v in reply.items():
            if k in _INFO_FIELDS:
                continue
            obs[k] = v if isinstance(v, list) else float(v)
        if obs.get("stairs_dist", 0.0) < 0:
            obs["stairs_dist"] = UNKNOWN_STAIRS_DIST
        return obs

    # ------------------------------------------------------------- GameEnv
    def reset(self) -> Observation:
        self.steps = 0
        depth = 1
        if self.curriculum is not None:
            depth = max(1, int(self.curriculum(self.episode)))
        self.start_depth = depth
        reply = self._send(
            f"reset {self.base_seed + self.episode} {self.hero} {self.challenges} {depth}"
        )
        self.episode += 1
        return self._to_obs(reply)

    def step(self, action: str) -> tuple[Observation, bool, dict[str, Any]]:
        self.steps += 1
        reply = self._send(f"act {action}")
        obs = self._to_obs(reply)
        done = bool(reply.get("done")) or self.steps >= self.max_steps
        info = {"depth": obs.get("depth", 1.0), "turns": reply.get("turns", 0),
                "won": bool(obs.get("won", 0.0)), "gear": reply.get("gear", "")}
        depth = int(obs.get("depth", 1))
        # only count floors reached from a real floor-1 start — a curriculum
        # episode that BEGINS on floor 4 hasn't achieved floor 4
        if self.start_depth == 1 and depth > self.best_depth:
            self.best_depth = depth
            self.best_gear = info["gear"]
        return obs, done, info

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "SPDRealEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
