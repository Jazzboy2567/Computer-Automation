"""A small DQN: function approximation instead of a lookup table.

Why: the tabular agent fragments — every unseen feature combination is a brand
new state with no knowledge carried over. A neural network generalizes across
the observation space (HP 13/20 informs HP 12/20), which is what large state
spaces like the real SPD need.

Deliberately dependency-free (pure NumPy): a 2-hidden-layer MLP with replay,
a target network, and Adam. Same interface as `QLearningAgent`, so `train()`
and `evaluate()` drive it unchanged — but it learns over the FULL numeric
observation (featurizer = identity), not a hand-compacted key subset.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np

from .env import Observation


class _MLP:
    """obs -> Q-values. He-init, ReLU, Adam. Small enough to train on CPU."""

    def __init__(self, n_in: int, n_out: int, hidden: int, rng: np.random.Generator):
        def he(shape):
            return rng.normal(0.0, np.sqrt(2.0 / shape[0]), size=shape).astype(np.float32)

        self.params = {
            "W1": he((n_in, hidden)), "b1": np.zeros(hidden, np.float32),
            "W2": he((hidden, hidden)), "b2": np.zeros(hidden, np.float32),
            "W3": he((hidden, n_out)), "b3": np.zeros(n_out, np.float32),
        }
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.params.items()}
        self._t = 0

    def forward(self, x: np.ndarray):
        p = self.params
        z1 = x @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0.0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0.0)
        q = a2 @ p["W3"] + p["b3"]
        return q, (x, z1, a1, z2, a2)

    def backward_step(self, cache, dq: np.ndarray, lr: float):
        """One Adam step from dLoss/dQ."""
        x, z1, a1, z2, a2 = cache
        p = self.params
        grads = {
            "W3": a2.T @ dq, "b3": dq.sum(0),
        }
        da2 = dq @ p["W3"].T
        dz2 = da2 * (z2 > 0)
        grads["W2"] = a1.T @ dz2; grads["b2"] = dz2.sum(0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (z1 > 0)
        grads["W1"] = x.T @ dz1; grads["b1"] = dz1.sum(0)

        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** self._t)
            vhat = v / (1 - b2 ** self._t)
            p[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def copy_from(self, other: "_MLP"):
        for k in self.params:
            self.params[k][:] = other.params[k]


class DQNAgent:
    """Drop-in replacement for QLearningAgent, backed by a neural net.

    Feature order is locked from the first observation seen; missing keys read
    as 0. Features are normalized by a running per-feature max magnitude so
    HP (0..20+), gold (0..1000+), and flags (0/1) share a scale.
    """

    def __init__(self, actions: list[str], seed: int = 0, hidden: int = 64,
                 lr: float = 1e-3, gamma: float = 0.99,
                 buffer_size: int = 50_000, batch_size: int = 64,
                 warmup: int = 500, learn_every: int = 4, sync_every: int = 1000):
        self.actions = list(actions)
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.warmup = warmup
        self.learn_every = learn_every
        self.sync_every = sync_every

        self._rng = random.Random(seed)
        self._nprng = np.random.default_rng(seed)
        self._keys: Optional[list[str]] = None
        self._array_keys: list[str] = []
        self._scale: Optional[np.ndarray] = None
        self._net: Optional[_MLP] = None
        self._target: Optional[_MLP] = None
        self._hidden = hidden
        self._buffer: deque = deque(maxlen=buffer_size)
        self._steps = 0
        self._updates = 0

    # ------------------------------------------------------------ features
    def _vec(self, obs: Observation) -> np.ndarray:
        if self._keys is None:
            # scalar fields feed the vector by name; array fields (e.g. the
            # egocentric `map` planes) are appended flat, in sorted key order
            self._keys = sorted(k for k, v in obs.items()
                                if not isinstance(v, (list, tuple, np.ndarray)))
            self._array_keys = sorted(k for k, v in obs.items()
                                      if isinstance(v, (list, tuple, np.ndarray)))
            n_in = len(self._keys) + sum(len(obs[k]) for k in self._array_keys)
            self._scale = np.ones(n_in, np.float32)
            self._net = _MLP(n_in, len(self.actions), self._hidden, self._nprng)
            self._target = _MLP(n_in, len(self.actions), self._hidden, self._nprng)
            self._target.copy_from(self._net)
        parts = [np.array([float(obs.get(k, 0.0)) for k in self._keys], np.float32)]
        for k in self._array_keys:
            parts.append(np.asarray(obs[k], np.float32))
        x = np.concatenate(parts) if len(parts) > 1 else parts[0]
        np.maximum(self._scale, np.abs(x), out=self._scale)   # running max magnitude
        return x / self._scale

    # ------------------------------------------------------------ interface
    @property
    def states_learned(self) -> int:
        return self._updates            # gradient updates stand in for "states"

    @property
    def Q(self) -> dict:
        """Serializable snapshot (joblib): weights + feature schema."""
        if self._net is None:
            return {}
        return {"keys": self._keys, "array_keys": self._array_keys, "scale": self._scale,
                "params": {k: v.copy() for k, v in self._net.params.items()},
                "actions": self.actions}

    @Q.setter
    def Q(self, snapshot: dict) -> None:
        if not snapshot:
            return
        self._keys = list(snapshot["keys"])
        self._array_keys = list(snapshot.get("array_keys", []))
        self._scale = np.asarray(snapshot["scale"], np.float32)
        n_in = len(self._scale)   # scalars + flattened array fields
        self._net = _MLP(n_in, len(self.actions), self._hidden, self._nprng)
        self._net.params = {k: np.asarray(v, np.float32) for k, v in snapshot["params"].items()}
        self._target = _MLP(n_in, len(self.actions), self._hidden, self._nprng)
        self._target.copy_from(self._net)

    def act(self, obs: Observation, epsilon: float) -> str:
        if self._rng.random() < epsilon:
            return self._rng.choice(self.actions)
        return self.policy(obs)

    def policy(self, obs: Observation) -> str:
        x = self._vec(obs)
        q, _ = self._net.forward(x[None, :])
        return self.actions[int(np.argmax(q[0]))]

    def learn(self, obs: Observation, action: str, reward: float,
              next_obs: Observation, done: bool) -> None:
        self._buffer.append((self._vec(obs), self.actions.index(action),
                             float(reward), self._vec(next_obs), bool(done)))
        self._steps += 1
        if len(self._buffer) < self.warmup or self._steps % self.learn_every:
            return

        batch = self._rng.sample(range(len(self._buffer)), min(self.batch_size, len(self._buffer)))
        xs = np.stack([self._buffer[i][0] for i in batch])
        acts = np.array([self._buffer[i][1] for i in batch])
        rews = np.array([self._buffer[i][2] for i in batch], np.float32)
        nxts = np.stack([self._buffer[i][3] for i in batch])
        dones = np.array([self._buffer[i][4] for i in batch], bool)

        # Double DQN: the ONLINE net selects the next action, the TARGET net
        # evaluates it. Vanilla `max` over the target net systematically
        # overestimates action values, which here let the policy collapse onto a
        # single over-valued action (throw_item ~ a safe no-op) and stop
        # descending. Decoupling selection from evaluation curbs that.
        next_acts = self._net.forward(nxts)[0].argmax(1)
        q_next, _ = self._target.forward(nxts)
        q_next_sel = q_next[np.arange(len(batch)), next_acts]
        targets = rews + np.where(dones, 0.0, self.gamma * q_next_sel)

        q, cache = self._net.forward(xs)
        dq = np.zeros_like(q)
        idx = np.arange(len(batch))
        # clipped TD error (Huber-style gradient) for stability
        td = np.clip(q[idx, acts] - targets, -1.0, 1.0)
        dq[idx, acts] = td / len(batch)
        self._net.backward_step(cache, dq, self.lr)

        self._updates += 1
        if self._updates % self.sync_every == 0:
            self._target.copy_from(self._net)
