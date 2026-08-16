"""Optional PyTorch/CUDA backend for the DQN — the same tiny MLP, but the batch
gradient step (the bottleneck: ~94% of training time is batch-64 matmuls) runs on
the GPU instead of pure-NumPy on one CPU core.

Design goals:
  * Interop: params serialize to the SAME numpy dict format the pure-NumPy net
    uses ({W1,b1,W2,b2,W3,b3} or dueling {..,Wv,bv,Wa,ba}), so existing
    policy*.joblib checkpoints load either way and the NumPy eval/diagnostic path
    still works on a GPU-trained policy.
  * One GPU round-trip per update: the whole Double-DQN target + forward/backward
    + Adam happens inside `train_double_dqn`, so we don't shuttle tensors per op.
  * Behavioural (not bit-exact) parity with the NumPy path: Huber/clipped-TD loss
    (delta=1 → gradient clamped to [-1,1], matching the NumPy `np.clip(td,-1,1)`),
    Adam(lr, 0.9, 0.999, 1e-8), importance-sampling weights, mean over the batch.

Import is guarded: this module is only imported when torch is actually installed
and the caller asks for the torch backend, so the default NumPy path has no new
dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cuda_available() -> bool:
    return torch.cuda.is_available()


class _MLPModule(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class _DuelingModule(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        adv = self.a(x)
        return v + adv - adv.mean(1, keepdim=True)


class TorchNet:
    """Wraps a torch module + Adam on a device, exposing the same surface the
    DQNAgent needs from the NumPy `_MLP`: `.params` (numpy dict, both get/set),
    `forward` (numpy in/out for the policy), `copy_from`, plus `train_double_dqn`
    which performs the whole batched update on-device."""

    def __init__(self, n_in: int, n_out: int, hidden: int, dueling: bool,
                 lr: float = 1e-3, device: Optional[str] = None, seed: int = 0):
        self.dueling = dueling
        self.n_in, self.n_out, self.hidden = n_in, n_out, hidden
        self.device = torch.device(device or ("cuda" if cuda_available() else "cpu"))
        g = torch.Generator(device="cpu").manual_seed(seed)
        # build on CPU with a seeded init, then move to device (He/kaiming to match
        # the NumPy net's variance)
        mod = _DuelingModule(n_in, n_out, hidden) if dueling else _MLPModule(n_in, n_out, hidden)
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    w = torch.empty_like(m.weight)
                    nn.init.kaiming_normal_(w, nonlinearity="relu", generator=g) \
                        if _init_supports_generator() else nn.init.kaiming_normal_(w, nonlinearity="relu")
                    m.weight.copy_(w)
                    m.bias.zero_()
        self.mod = mod.to(self.device)
        self.opt = torch.optim.Adam(self.mod.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # ---- interop: expose weights in the NumPy net's {W,b} layout -------------
    @property
    def params(self) -> dict:
        p = {}
        sd = self.mod.state_dict()
        if self.dueling:
            names = [("fc1", "W1", "b1"), ("fc2", "W2", "b2"), ("v", "Wv", "bv"), ("a", "Wa", "ba")]
        else:
            names = [("fc1", "W1", "b1"), ("fc2", "W2", "b2"), ("fc3", "W3", "b3")]
        for lin, wk, bk in names:
            p[wk] = sd[f"{lin}.weight"].detach().cpu().numpy().T.astype(np.float32)  # (out,in)->(in,out)
            p[bk] = sd[f"{lin}.bias"].detach().cpu().numpy().astype(np.float32)
        return p

    @params.setter
    def params(self, p: dict) -> None:
        if self.dueling:
            names = [("fc1", "W1", "b1"), ("fc2", "W2", "b2"), ("v", "Wv", "bv"), ("a", "Wa", "ba")]
        else:
            names = [("fc1", "W1", "b1"), ("fc2", "W2", "b2"), ("fc3", "W3", "b3")]
        sd = self.mod.state_dict()
        for lin, wk, bk in names:
            sd[f"{lin}.weight"].copy_(torch.as_tensor(np.asarray(p[wk], np.float32).T, device=self.device))
            sd[f"{lin}.bias"].copy_(torch.as_tensor(np.asarray(p[bk], np.float32), device=self.device))
        self.mod.load_state_dict(sd)

    def copy_from(self, other: "TorchNet") -> None:
        self.mod.load_state_dict(other.mod.state_dict())

    # ---- policy forward (numpy in/out, small batches) ------------------------
    def forward(self, x: np.ndarray):
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x, np.float32), device=self.device)
            q = self.mod(t)
        return q.cpu().numpy(), None

    # ---- the whole Double-DQN batch update, on-device ------------------------
    def train_double_dqn(self, xs, acts, rews, nxts, dones, weights, gamma, target: "TorchNet"):
        """Returns td_raw (numpy) for priority updates. Online net selects the next
        action, target net evaluates it; clipped-TD (Huber) loss x IS weights."""
        dev = self.device
        xs_t = torch.as_tensor(xs, device=dev)
        nx_t = torch.as_tensor(nxts, device=dev)
        acts_t = torch.as_tensor(np.asarray(acts, np.int64), device=dev)
        rews_t = torch.as_tensor(np.asarray(rews, np.float32), device=dev)
        dones_t = torch.as_tensor(np.asarray(dones, np.float32), device=dev)
        w_t = torch.as_tensor(np.asarray(weights, np.float32), device=dev)

        with torch.no_grad():
            next_acts = self.mod(nx_t).argmax(1)                      # online selects
            q_next = target.mod(nx_t).gather(1, next_acts[:, None]).squeeze(1)  # target evaluates
            targets = rews_t + (1.0 - dones_t) * gamma * q_next

        q = self.mod(xs_t)
        q_sel = q.gather(1, acts_t[:, None]).squeeze(1)
        td = q_sel - targets
        # Huber (delta=1) gives a gradient clamped to [-1,1], matching the NumPy
        # clipped-TD; weight by IS weights, mean over the batch.
        per = F.smooth_l1_loss(q_sel, targets, reduction="none")
        loss = (per * w_t).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return td.detach().cpu().numpy()


def _init_supports_generator() -> bool:
    # kaiming_normal_ gained a `generator=` kwarg in newer torch; degrade gracefully
    try:
        import inspect
        return "generator" in inspect.signature(nn.init.kaiming_normal_).parameters
    except Exception:
        return False
