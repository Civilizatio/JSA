"""Joint negative sampler for stage-3 Perceptual JSA training."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor
import torch.nn as nn

from src.base.base_sampler import BaseSampler


class JointReplayBuffer:
    """A simple replay buffer for storing (x, h) pairs from previous negative sampling rounds.

    Arguments
    ---------
    capacity: int
        Maximum number of (x, h) pairs to store in the buffer. If <= 0, the buffer is disabled and no samples will be stored.
    store_on_cpu: bool
        If True, stored samples will be kept on CPU to save GPU memory. Samples will be moved to the appropriate device when sampled. If False, samples will be stored on the same device as they were pushed.
    """

    def __init__(self, capacity: int = 0, store_on_cpu: bool = True):
        self.capacity = int(capacity)
        self.store_on_cpu = bool(store_on_cpu)
        self.size = 0
        self.ptr = 0
        self.x_buffer = None
        self.h_buffer = None

    def _maybe_to_store_device(self, x: Tensor) -> Tensor:
        if self.store_on_cpu:
            return x.detach().cpu()
        return x.detach().clone()

    def push(self, x: Tensor, h: Tensor) -> None:
        if self.capacity <= 0:
            return
        x_store = self._maybe_to_store_device(x)
        h_store = self._maybe_to_store_device(h)

        if self.x_buffer is None:
            self.x_buffer = torch.empty(
                (self.capacity, *x_store.shape[1:]),
                dtype=x_store.dtype,
                device=x_store.device,
            )
            self.h_buffer = torch.empty(
                (self.capacity, *h_store.shape[1:]),
                dtype=h_store.dtype,
                device=h_store.device,
            )

        batch_size = x_store.shape[0]
        for i in range(batch_size):
            self.x_buffer[self.ptr] = x_store[i]
            self.h_buffer[self.ptr] = h_store[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device) -> Optional[tuple[Tensor, Tensor]]:
        if self.size == 0:
            return None
        idx = torch.randint(0, self.size, (batch_size,))
        x = self.x_buffer[idx].to(device)
        h = self.h_buffer[idx].to(device)
        return x, h

    def state_dict(self):
        if self.capacity <= 0 or self.size == 0:
            return {"capacity": self.capacity, "size": 0}
        return {
            "capacity": self.capacity,
            "size": self.size,
            "ptr": self.ptr,
            "x_buffer": self.x_buffer[: self.size].cpu(),
            "h_buffer": self.h_buffer[: self.size].cpu(),
            "store_on_cpu": self.store_on_cpu,
        }

    def load_state_dict(self, state):
        self.capacity = int(state.get("capacity", self.capacity))
        self.size = int(state.get("size", 0))
        self.ptr = int(state.get("ptr", 0))
        self.store_on_cpu = bool(state.get("store_on_cpu", self.store_on_cpu))
        x_buffer = state.get("x_buffer")
        h_buffer = state.get("h_buffer")
        if x_buffer is None or h_buffer is None or self.size == 0:
            self.x_buffer = None
            self.h_buffer = None
            return
        self.x_buffer = torch.empty(
            (self.capacity, *x_buffer.shape[1:]), dtype=x_buffer.dtype
        )
        self.h_buffer = torch.empty(
            (self.capacity, *h_buffer.shape[1:]), dtype=h_buffer.dtype
        )
        self.x_buffer[: self.size] = x_buffer
        self.h_buffer[: self.size] = h_buffer


class JointNegativeSampler(BaseSampler):
    def __init__(
        self,
        joint_model,
        proposal_model,
        h_sampler,
        x_sampler,
        ncg_sampler=None,
        num_rounds: int = 1,
        replay_buffer_size: int = 0,
        replay_prob: float = 0.95,
        init_noise_std: float = 0.01,
        clamp_range: Optional[Sequence[float]] = (-1.0, 1.0),
    ):
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.h_sampler = h_sampler
        self.x_sampler = x_sampler
        self.ncg_sampler = ncg_sampler
        self.num_rounds = int(num_rounds)
        self.replay_prob = float(replay_prob)
        self.init_noise_std = float(init_noise_std)
        self.clamp_range = tuple(clamp_range) if clamp_range is not None else None
        self.replay_buffer = JointReplayBuffer(capacity=replay_buffer_size)

    def to(self, device):
        if hasattr(self.h_sampler, "to"):
            self.h_sampler.to(device)
        if hasattr(self.x_sampler, "to"):
            self.x_sampler.to(device)
        if self.ncg_sampler is not None and hasattr(self.ncg_sampler, "to"):
            self.ncg_sampler.to(device)
        return self

    def _clamp(self, x: Tensor) -> Tensor:
        if self.clamp_range is None:
            return x
        return x.clamp(self.clamp_range[0], self.clamp_range[1])

    def _initialize_state(self, x_real: Tensor, h_pos: Optional[Tensor] = None):
        batch_size = x_real.shape[0]
        replay = self.replay_buffer.sample(batch_size, x_real.device)
        use_replay = (
            replay is not None
            and torch.rand(1, device=x_real.device).item() < self.replay_prob
        )

        if use_replay:
            x_init, h_init = replay
        else:
            x_init = x_real + self.init_noise_std * torch.randn_like(x_real)
            x_init = self._clamp(x_init)
            if h_pos is not None:
                if h_pos.dim() == 5 and h_pos.shape[1] == 1:
                    h_init = h_pos[:, 0].detach().clone()
                else:
                    h_init = h_pos.detach().clone()
            else:
                h_init = self.proposal_model.sample_latent(x_real).squeeze(1)

        return x_init.detach(), h_init.detach()

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        x_new = self.x_sampler.sample(x, h)
        h_new = self.h_sampler.sample(
            x_new,
            idx=None,
            num_steps=10,
            parallel=False,
            return_all=False,
            strategy="none",
        )
        if h_new.dim() == 5 and h_new.shape[1] == 1:
            h_new = h_new[:, 0]
        if self.ncg_sampler is not None:
            h_new = self.ncg_sampler.sample(x_new, h_new)
        return x_new.detach(), h_new.detach()

    def sample(self, x_real: Tensor, h_pos: Optional[Tensor] = None):
        x_cur, h_cur = self._initialize_state(x_real, h_pos=h_pos)
        for _ in range(self.num_rounds):
            x_cur, h_cur = self.step(x_cur, h_cur)

        self.replay_buffer.push(x_cur, h_cur)
        return x_cur.detach(), h_cur.detach()

    def state_dict(self):
        return {"replay_buffer": self.replay_buffer.state_dict()}

    def load_state_dict(self, state):
        if "replay_buffer" in state:
            self.replay_buffer.load_state_dict(state["replay_buffer"])


__all__ = ["JointNegativeSampler", "JointReplayBuffer"]
