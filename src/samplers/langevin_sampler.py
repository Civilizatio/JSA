# src/samplers/langevin_sampler.py
"""Langevin dynamics sampler for the continuous observation variable ``x``."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn as nn


class LangevinSampler:
    """Sample ``x`` with fixed ``h`` using overdamped Langevin dynamics.

    The sampler follows the update

    ``x_{t+1} = x_t - 0.5 * eps^2 * grad_x d(x_t, h) + eps * noise``

    where ``d`` is the distortion term of the joint energy.
    """

    def __init__(
        self,
        joint_model: nn.Module,
        step_size: float = 0.05,
        noise_scale: float = 1.0,
        num_steps: int = 10,
        clamp_range: Optional[Sequence[float]] = (-1.0, 1.0),
        detach_between_steps: bool = True,
    ):
        self.joint_model = joint_model
        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)
        self.num_steps = int(num_steps)
        self.clamp_range = tuple(clamp_range) if clamp_range is not None else None
        self.detach_between_steps = bool(detach_between_steps)

    def to(self, device: torch.device):
        # Stateless sampler. Kept for interface symmetry with MISampler.
        return self

    def _clamp(self, x: Tensor) -> Tensor:
        if self.clamp_range is None:
            return x
        return x.clamp(self.clamp_range[0], self.clamp_range[1])

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        x_in = x.detach() if self.detach_between_steps else x
        x_in = x_in.requires_grad_(True)
        distortion = self.joint_model.distortion(x_in, h)
        grad = torch.autograd.grad(distortion.sum(), x_in, only_inputs=True)[0]
        noise = torch.randn_like(x_in) * self.step_size * self.noise_scale
        x_next = x_in - 0.5 * (self.step_size**2) * grad + noise
        x_next = self._clamp(x_next)
        return x_next.detach() if self.detach_between_steps else x_next

    def sample(
        self,
        x_init: Tensor,
        h: Tensor,
        num_steps: Optional[int] = None,
        return_all: bool = False,
    ):
        steps = self.num_steps if num_steps is None else int(num_steps)
        x = x_init
        trajectory: List[Tensor] = []
        for _ in range(steps):
            x = self.step(x, h)
            if return_all:
                trajectory.append(x)
        if return_all:
            return torch.stack(trajectory, dim=1)
        return x


__all__ = ["LangevinSampler"]
