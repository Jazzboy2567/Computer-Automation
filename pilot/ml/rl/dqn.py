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
                 warmup: int = 500, learn_every: int = 4, sync_every: int = 1000,
                 prioritized: bool = False, prio_alpha: float = 0.6,
                 prio_beta: float = 0.4, prio_eps: float = 1e-3):
        self.actions = list(actions)
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.warmup = warmup
        self.learn_every = learn_every
        self.sync_every = sync_every
        # Prioritized replay: sample transitions in proportion to how SURPRISING
        # they were (|TD error|), not uniformly. The idea was to make the agent
        # train more on the rare floor 2-3 deaths that a 50k uniform buffer
        # drowns. In practice, at alpha=0.6/beta=0.4 it DESTABILISED learning —
        # over-focusing on the big-negative-TD death transitions collapsed the
        # policy back to floor 1 (return -35 vs uniform +100+). So it is OFF by
        # default; kept as an option to revisit with gentler settings (lower
        # alpha, annealed beta). alpha = prioritization strength (0 = uniform),
        # beta = importance-sampling correction for the bias it introduces.
        self.prioritized = prioritized
        self.prio_alpha = prio_alpha
        self.prio_beta = prio_beta
        self.prio_eps = prio_eps

        self._rng = random.Random(seed)
        self._nprng = np.random.default_rng(seed)
        self._keys: Optional[list[str]] = None
        self._array_keys: list[str] = []
        self._scale: Optional[np.ndarray] = None
        self._net: Optional[_MLP] = None
        self._target: Optional[_MLP] = None
        self._hidden = hidden
        # circular replay buffer as preallocated arrays (needed so priorities stay
        # aligned with transitions and sampling/updates are O(1) indexed) —
        # allocated lazily on the first learn, once the feature width is known
        self._cap = buffer_size
        self._obs_buf: Optional[np.ndarray] = None
        self._nobs_buf: Optional[np.ndarray] = None
        self._act_buf: Optional[np.ndarray] = None
        self._rew_buf: Optional[np.ndarray] = None
        self._done_buf: Optional[np.ndarray] = None
        self._prio_buf: Optional[np.ndarray] = None
        self._pos = 0
        self._size = 0
        self._max_prio = 1.0
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

    def _alloc(self, d: int) -> None:
        if self._obs_buf is not None:
            return
        cap = self._cap
        self._obs_buf = np.zeros((cap, d), np.float32)
        self._nobs_buf = np.zeros((cap, d), np.float32)
        self._act_buf = np.zeros(cap, np.int64)
        self._rew_buf = np.zeros(cap, np.float32)
        self._done_buf = np.zeros(cap, bool)
        self._prio_buf = np.zeros(cap, np.float32)

    def learn(self, obs: Observation, action: str, reward: float,
              next_obs: Observation, done: bool) -> None:
        x, nx = self._vec(obs), self._vec(next_obs)
        self._alloc(len(x))
        i = self._pos
        self._obs_buf[i] = x
        self._nobs_buf[i] = nx
        self._act_buf[i] = self.actions.index(action)
        self._rew_buf[i] = reward
        self._done_buf[i] = done
        self._prio_buf[i] = self._max_prio      # new transitions get top priority
        self._pos = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

        self._steps += 1
        if self._size < self.warmup or self._steps % self.learn_every:
            return

        n, k = self._size, min(self.batch_size, self._size)
        if self.prioritized:
            scaled = self._prio_buf[:n] ** self.prio_alpha
            probs = scaled / scaled.sum()
            idx = self._nprng.choice(n, k, p=probs)
            weights = (n * probs[idx]) ** (-self.prio_beta)
            weights = (weights / weights.max()).astype(np.float32)   # IS correction
        else:
            idx = self._nprng.choice(n, k, replace=False)
            weights = np.ones(k, np.float32)

        xs, acts = self._obs_buf[idx], self._act_buf[idx]
        rews, nxts, dones = self._rew_buf[idx], self._nobs_buf[idx], self._done_buf[idx]

        # Double DQN: the ONLINE net selects the next action, the TARGET net
        # evaluates it. Vanilla `max` over the target net systematically
        # overestimates action values, which here let the policy collapse onto a
        # single over-valued action (throw_item ~ a safe no-op) and stop
        # descending. Decoupling selection from evaluation curbs that.
        next_acts = self._net.forward(nxts)[0].argmax(1)
        q_next, _ = self._target.forward(nxts)
        q_next_sel = q_next[np.arange(k), next_acts]
        targets = rews + np.where(dones, 0.0, self.gamma * q_next_sel)

        q, cache = self._net.forward(xs)
        ar = np.arange(k)
        td_raw = q[ar, acts] - targets
        # the fresh |TD error| becomes each transition's new priority
        self._prio_buf[idx] = np.abs(td_raw) + self.prio_eps
        self._max_prio = max(self._max_prio, float(self._prio_buf[idx].max()))

        dq = np.zeros_like(q)
        # clipped TD error (Huber-style gradient) x IS weight, for stability
        dq[ar, acts] = (np.clip(td_raw, -1.0, 1.0) * weights) / k
        self._net.backward_step(cache, dq, self.lr)

        self._updates += 1
        if self._updates % self.sync_every == 0:
            self._target.copy_from(self._net)
