# src/samplers/langevin_sampler.py
"""Langevin / MALA sampler for the continuous observation variable ``x``."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from src.base.base_sampler import BaseSampler


class LangevinSampler(BaseSampler):
    """Sample ``x`` with fixed ``h`` using overdamped Langevin dynamics.

    The sampler follows the MALA update

    ``x_{t+1} = x_t - 0.5 * alpha^2 * grad_x d(x_t, h) + alpha * eps``

    where ``alpha = step_size * noise_scale`` is the effective step size and
    ``d`` is the distortion term of the joint energy.  The gradient coefficient
    and the noise variance are both determined by ``alpha``, keeping the
    proposal distribution self-consistent:

    ``q(x' | x) = N(x' | x - 0.5*alpha^2*grad_d(x), alpha^2 * I)``

    When ``use_mala`` is True, each proposal is followed by a Metropolis-Hastings
    accept/reject correction using the exact log-ratio of this Gaussian kernel,
    yielding a complete MALA step.
    """

    def __init__(
        self,
        joint_model: nn.Module,
        step_size: float = 0.05,
        noise_scale: float = 1.0,
        num_steps: int = 10,
        clamp_range: Optional[Sequence[float]] = (-1.0, 1.0),
        detach_between_steps: bool = True,
        use_mala: bool = True,
        acceptance_log_clip: float = -100.0,
    ):
        self.joint_model = joint_model
        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)
        self.num_steps = int(num_steps)
        self.clamp_range = tuple(clamp_range) if clamp_range is not None else None
        self.detach_between_steps = bool(detach_between_steps)
        self.use_mala = bool(use_mala)
        self.acceptance_log_clip = float(acceptance_log_clip)
        self._accept_cnt = torch.zeros(1)
        self._total_cnt = torch.zeros(1)

    def to(self, device: torch.device):
        # Stateless sampler. Kept for interface symmetry with MISampler.
        self._accept_cnt = self._accept_cnt.to(device)
        self._total_cnt = self._total_cnt.to(device)
        return self

    def _clamp(self, x: Tensor) -> Tensor:
        if self.clamp_range is None:
            return x
        return x.clamp(self.clamp_range[0], self.clamp_range[1])

    @staticmethod
    def _as_batch_vector(t: Tensor) -> Tensor:
        if t.dim() == 1:
            return t
        return t.view(t.shape[0], -1).mean(dim=1)

    def reset_acceptance_stats(self):
        device = self._accept_cnt.device
        self._accept_cnt = torch.zeros(1, device=device)
        self._total_cnt = torch.zeros(1, device=device)

    def get_acceptance_rate(self) -> float:
        if self._total_cnt.item() <= 0:
            return 1.0
        return (self._accept_cnt / self._total_cnt).item()

    def _distortion_and_grad(self, x: Tensor, h: Tensor):
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            out = self.joint_model.distortion(x_req, h)
            distortion = self._as_batch_vector(out.distortion)
            grad = torch.autograd.grad(
                distortion.sum(),
                x_req,
                only_inputs=True,
            )[0]
        return distortion.detach(), grad.detach()

    @staticmethod
    def _gaussian_log_prob(x: Tensor, mean: Tensor, std: float) -> Tensor:
        var = max(std, 1e-8) ** 2
        diff = x - mean
        log_prob = -0.5 * (diff * diff) / var
        return log_prob.view(log_prob.shape[0], -1).sum(dim=1)

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        x_cur = x.detach() if self.detach_between_steps else x
        old_distortion, grad_old = self._distortion_and_grad(x_cur, h)

        # alpha = step_size * noise_scale is the effective step size.
        # The MALA proposal is N(x' | x - alpha^2/2 * grad_d, alpha^2 * I).
        # Both the gradient coefficient and the noise std must use the same alpha
        # so that the MH log-ratio is computed against the correct proposal density.
        noise_std = self.step_size * self.noise_scale
        mean_forward = x_cur - 0.5 * (noise_std**2) * grad_old
        proposal = mean_forward + torch.randn_like(x_cur) * noise_std
        proposal = self._clamp(proposal)

        if not self.use_mala:
            return proposal.detach() if self.detach_between_steps else proposal

        new_distortion, grad_new = self._distortion_and_grad(proposal, h)
        mean_reverse = proposal - 0.5 * (noise_std**2) * grad_new

        log_pi_old = -old_distortion
        log_pi_new = -new_distortion
        log_q_forward = self._gaussian_log_prob(proposal, mean_forward, noise_std)
        log_q_reverse = self._gaussian_log_prob(x_cur, mean_reverse, noise_std)

        log_accept = (log_pi_new + log_q_reverse) - (log_pi_old + log_q_forward)
        
        # 使用对数空间的均匀分布判定接受率，免去 exp 带来的数值截断或溢出问题
        log_u = torch.log(torch.rand_like(log_accept).clamp_min(1e-8))
        accept = log_u < log_accept

        self._accept_cnt += accept.float().sum()
        self._total_cnt += torch.tensor(float(accept.numel()), device=accept.device)

        accept_view = accept.view(-1, *([1] * (proposal.dim() - 1)))
        x_next = torch.where(accept_view, proposal, x_cur)
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
