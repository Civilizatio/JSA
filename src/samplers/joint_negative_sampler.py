"""Joint negative sampler for stage-3 Perceptual JSA training."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from src.base.base_sampler import BaseSampler


class JointReplayBuffer:
    """Replay buffer for persistent negative samples ``(x, h)``."""

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

    def _allocate(self, x_store: Tensor, h_store: Tensor):
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

    def push(self, x: Tensor, h: Tensor) -> None:
        if self.capacity <= 0:
            return

        x_store = self._maybe_to_store_device(x)
        h_store = self._maybe_to_store_device(h)

        if self.x_buffer is None:
            self._allocate(x_store, h_store)

        batch_size = x_store.shape[0]
        write_idx = (
            torch.arange(batch_size, device=x_store.device, dtype=torch.long) + self.ptr
        ) % self.capacity
        self.x_buffer[write_idx] = x_store
        self.h_buffer[write_idx] = h_store
        self.ptr = int((self.ptr + batch_size) % self.capacity)
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, device) -> Optional[tuple[Tensor, Tensor]]:
        if self.size == 0:
            return None

        idx = torch.randint(0, self.size, (batch_size,), device=self.x_buffer.device)
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
        h_num_steps: int = 10,
        h_parallel: bool = False,
        h_strategy: str = "none",
        use_ncg_after_h: bool = True,
        x_num_steps: Optional[int] = None,
        replay_store_on_cpu: bool = True,
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
        self.h_num_steps = int(h_num_steps)
        self.h_parallel = bool(h_parallel)
        self.h_strategy = str(h_strategy)
        self.use_ncg_after_h = bool(use_ncg_after_h)
        self.x_num_steps = None if x_num_steps is None else int(x_num_steps)
        self.replay_buffer = JointReplayBuffer(
            capacity=replay_buffer_size,
            store_on_cpu=bool(replay_store_on_cpu),
        )

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

    @staticmethod
    def _sanitize_h(h: Tensor) -> Tensor:
        if h.dim() == 5 and h.shape[1] == 1:
            h = h[:, 0]
        return h.detach().clone().float()

    def _initialize_state(self, x_real: Tensor, h_pos: Optional[Tensor] = None):
        batch_size = x_real.shape[0]

        x_fresh = x_real + self.init_noise_std * torch.randn_like(x_real)
        x_fresh = self._clamp(x_fresh)

        if h_pos is not None:
            h_fresh = self._sanitize_h(h_pos)
        else:
            h_fresh = self.proposal_model.sample_latent(x_real).squeeze(1)
            h_fresh = self._sanitize_h(h_fresh)

        replay = self.replay_buffer.sample(batch_size, x_real.device)
        if replay is None or self.replay_prob <= 0.0:
            return x_fresh.detach(), h_fresh.detach()

        x_replay, h_replay = replay
        use_replay_mask = torch.rand(batch_size, device=x_real.device) < self.replay_prob
        if not use_replay_mask.any():
            return x_fresh.detach(), h_fresh.detach()

        x_init = x_fresh.clone()
        h_init = h_fresh.clone()
        x_init[use_replay_mask] = x_replay[use_replay_mask]
        h_init[use_replay_mask] = h_replay[use_replay_mask]
        return x_init.detach(), h_init.detach()

    def _sample_x(self, x: Tensor, h: Tensor) -> Tensor:
        if self.x_sampler is None:
            raise RuntimeError("JointNegativeSampler requires an x_sampler in stage-3.")
        kwargs = {}
        if self.x_num_steps is not None:
            kwargs["num_steps"] = self.x_num_steps
        x_new = self.x_sampler.sample(x, h, **kwargs)
        return self._clamp(x_new).detach()

    def _sample_h(self, x: Tensor) -> Tensor:
        h_new = self.h_sampler.sample(
            x,
            idx=None,
            num_steps=self.h_num_steps,
            parallel=self.h_parallel,
            return_all=False,
            strategy=self.h_strategy,
        )
        h_new = self._sanitize_h(h_new)
        if self.use_ncg_after_h and self.ncg_sampler is not None:
            h_new = self.ncg_sampler.sample(x, h_new)
            h_new = self._sanitize_h(h_new)
        return h_new.detach()

    def step(self, x: Tensor, h: Tensor):
        x_new = self._sample_x(x, h)
        h_new = self._sample_h(x_new)
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
