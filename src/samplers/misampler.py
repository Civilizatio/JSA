# src/samplers/misampler.py
"""Metropolis Independence Sampler (MIS) for latent-variable JSA.

Compared with the original implementation, this version keeps the same public API while
adding two practical improvements:

1. it can fall back to a generic energy-model path when the joint model no longer has the
   Gaussian-specific ``sigma`` shortcut;
2. the parallel MIS path now works for any joint model that implements
   ``log_joint_prob_multiple_samples``.
"""

from __future__ import annotations

import warnings

import torch
import torch.distributed as dist

from src.base.base_sampler import BaseSampler


class MISampler(BaseSampler):
    """Metropolis Independence Sampler (MIS) for ``h ~ p(h | x)``.
    
    Arguments:
        joint_model: The joint model defining p(x, h).
        proposal_model: The proposal model defining q(h | x).
        use_cache: Whether to use caching for previously sampled latent variables.
        dataset_size: The total number of data points (required if use_cache=True).
        element_shape: The shape of each latent variable sample (required if use_cache=True).
        cache_dtype: The data type to use for caching latent variables (optional).
        prefer_fast_acceptance: Whether to use the faster acceptance probability calculation when possible.
        prefer_parallel_sampling: Whether to prefer parallel sampling when possible. 
        
        NOTE: `prefer_parallel_sampling` is the capability flag for the parallel sampling path, but the actual use of parallel sampling also depends on the `parallel` argument in the `sample` method. 
        Setting `prefer_parallel_sampling=False` will disable the parallel sampling path even if `parallel=True` is passed to `sample()`, while setting it to `True` will allow parallel sampling when `parallel=True` is used in `sample()`.
    """

    def __init__(
        self,
        joint_model,
        proposal_model,
        use_cache: bool = False,
        dataset_size=None,
        element_shape=None,
        cache_dtype=None,
        prefer_fast_acceptance: bool = True,
        prefer_parallel_sampling: bool = True,
    ):
        super().__init__()
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.dataset_size = dataset_size
        self.element_shape = element_shape
        self.prefer_fast_acceptance = bool(prefer_fast_acceptance)
        self.prefer_parallel_sampling = bool(prefer_parallel_sampling)

        if cache_dtype is not None:
            self.cache_dtype = cache_dtype
        else:
            self.cache_dtype = self._infer_cache_dtype()

        self.use_cache = use_cache
        self.reset_acceptance_stats()

    def _infer_cache_dtype(self):
        try:
            num_categories = self.joint_model.num_categories
            if isinstance(num_categories, list):
                codebook_size = max(num_categories)
            elif isinstance(num_categories, int):
                codebook_size = num_categories
            else:
                raise ValueError(
                    f"Unsupported type for num_categories: {type(num_categories)}"
                )

            if codebook_size <= torch.iinfo(torch.int8).max:
                return torch.int8
            if codebook_size <= torch.iinfo(torch.int16).max:
                return torch.int16
            if codebook_size <= torch.iinfo(torch.int32).max:
                return torch.int32
            return torch.int64
        except Exception:
            return torch.int64

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value: bool):
        self._use_cache = value
        if value and not hasattr(self, "cache"):
            self._init_cache(
                self.dataset_size,
                self.proposal_model.num_latent_vars,
                self.element_shape,
            )

    @torch.no_grad()
    def _init_cache(self, dataset_size, num_latent_vars, element_shape):
        cache_shape = (
            (dataset_size, *element_shape, num_latent_vars)
            if element_shape is not None
            else (dataset_size, num_latent_vars)
        )
        device = next(self.proposal_model.parameters()).device
        self.cache = torch.full(
            cache_shape,
            int(-1),
            dtype=self.cache_dtype,
            device=device,
        )
        self.updated_mask = torch.zeros(dataset_size, dtype=torch.bool, device=device)

    def to(self, device):
        self._accept_cnt = self._accept_cnt.to(device)
        self._total_cnt = self._total_cnt.to(device)
        if self.use_cache:
            self.cache = self.cache.to(device)
            self.updated_mask = self.updated_mask.to(device)
        return self

    @torch.no_grad()
    def _init_h_old(self, idx, h_new):
        if self.use_cache and idx is not None:
            h_old = self.cache[idx]
            h_old = h_old.unsqueeze(1).expand_as(h_new).float()
            uninitialized_mask = (h_old[:, 0:1, ...] == -1).all(
                dim=tuple(range(2, h_old.dim()))
            )
            if uninitialized_mask.any():
                mask_indices = torch.nonzero(uninitialized_mask.squeeze(1), as_tuple=True)[0]
                if len(mask_indices) > 0:
                    h_old[mask_indices] = h_new[mask_indices].clone()
        else:
            h_old = h_new.clone()
        return h_old

    def reset_acceptance_stats(self):
        device = next(self.proposal_model.parameters()).device
        self._accept_cnt = torch.zeros(1, device=device)
        self._total_cnt = torch.zeros(1, device=device)

    def _can_use_fast_acceptance(self):
        return (
            self.prefer_fast_acceptance
            and hasattr(self.joint_model, "log_joint_prob_diff")
            and hasattr(self.proposal_model, "log_conditional_prob_diff")
        )

    @torch.no_grad()
    def _cal_acceptance_prob(self, x, h_new, h_old, h_logits=None):
        log_p_new = self.joint_model.log_joint_prob_multiple_samples(x, h_new)
        log_p_old = self.joint_model.log_joint_prob_multiple_samples(x, h_old)

        if h_logits is not None:
            log_q_new = self.proposal_model.log_prob_from_logits(h_new, h_logits)
            log_q_old = self.proposal_model.log_prob_from_logits(h_old, h_logits)
        else:
            log_q_new = self.proposal_model.log_conditional_prob(h_new, x)
            log_q_old = self.proposal_model.log_conditional_prob(h_old, x)

        if (
            torch.isnan(log_p_new).any()
            or torch.isnan(log_p_old).any()
            or torch.isnan(log_q_new).any()
            or torch.isnan(log_q_old).any()
        ):
            raise ValueError("NaN encountered in MIS acceptance probability calculation.")
        if (
            torch.isinf(log_p_new).any()
            or torch.isinf(log_p_old).any()
            or torch.isinf(log_q_new).any()
            or torch.isinf(log_q_old).any()
        ):
            raise ValueError("Inf encountered in MIS acceptance probability calculation.")

        log_accept = (log_p_new + log_q_old) - (log_p_old + log_q_new)
        return torch.exp(torch.clamp(log_accept, max=0.0, min=-100.0))

    @torch.no_grad()
    def _cal_acceptance_prob_faster(self, x, h_new, h_old, h_logits=None):
        log_p_diff = self.joint_model.log_joint_prob_diff(x, h_new, h_old)
        log_q_diff = self.proposal_model.log_conditional_prob_diff(
            x, h_new, h_old, h_logits
        )
        log_accept = log_p_diff - log_q_diff
        return torch.exp(torch.clamp(log_accept, max=0.0, min=-100.0))

    def _sample_block_mask(self, h_shape, strategy="row"):
        mask = torch.zeros(
            h_shape,
            dtype=torch.bool,
            device=next(self.proposal_model.parameters()).device,
        )

        if strategy == "row":
            row = torch.randint(0, h_shape[2], (1,)).item()
            mask[:, :, row, :, :] = True
        elif strategy == "column":
            col = torch.randint(0, h_shape[3], (1,)).item()
            mask[:, :, :, col, :] = True
        elif strategy == "block":
            patch_size = 2
            row = torch.randint(0, h_shape[2] - patch_size + 1, (1,)).item()
            col = torch.randint(0, h_shape[3] - patch_size + 1, (1,)).item()
            mask[:, :, row : row + patch_size, col : col + patch_size, :] = True
        elif strategy == "none":
            mask[:] = True
        else:
            raise ValueError(f"Unsupported block sampling strategy: {strategy}")
        return mask

    @torch.no_grad()
    def step(self, x, idx=None, h_old=None, strategy="none"):
        if h_old is None:
            h_old = self._init_h_old(idx, self.proposal_model.sample_latent(x))

        h_new_full = self.proposal_model.sample_latent(x)
        block_mask = self._sample_block_mask(h_new_full.shape, strategy=strategy)
        h_new = h_old.clone()
        h_new[block_mask] = h_new_full[block_mask]

        if self._can_use_fast_acceptance():
            accept_prob = self._cal_acceptance_prob_faster(x, h_new, h_old)
        else:
            accept_prob = self._cal_acceptance_prob(x, h_new, h_old)

        accept = (torch.rand_like(accept_prob) < accept_prob).float()
        self._accept_cnt += accept.sum()
        self._total_cnt += accept.numel()

        accept = accept.view(accept.shape + (1,) * (h_new.dim() - accept.dim()))
        h_next = accept * h_new + (1 - accept) * h_old
        h_next = h_next.round()

        if self.use_cache and idx is not None:
            rand_indices = torch.randint(0, h_next.shape[1], (h_next.shape[0],), device=h_next.device)
            h_next_selected = h_next[torch.arange(h_next.shape[0], device=h_next.device), rand_indices]
            self.cache[idx] = h_next_selected.detach().to(self.cache.dtype)
            self.updated_mask[idx] = True

        return h_next

    @torch.no_grad()
    def sample(
        self,
        x,
        idx=None,
        num_steps: int = 1,
        parallel: bool = False,
        return_all: bool = False,
        strategy: str = "none",
    ):
        if parallel:
            if strategy != "none":
                warnings.warn(
                    "Parallel MIS does not support block-wise proposals; falling back to full proposals."
                )
            return self._sample_parallel(x, idx=idx, num_steps=num_steps, return_all=return_all)

        if return_all:
            h_list = []
        h_old = None
        for _ in range(num_steps):
            h_old = self.step(x, idx=idx, h_old=h_old, strategy=strategy)
            if return_all:
                h_list.append(h_old)
        if return_all:
            return torch.cat(h_list, dim=1)
        return h_old

    @torch.no_grad()
    def _sample_parallel(self, x, idx=None, num_steps: int = 1, return_all: bool = False):
        if not self.prefer_parallel_sampling:
            return self.sample(x, idx=idx, num_steps=num_steps, parallel=False, return_all=return_all)

        h_new_all, h_logits = self.proposal_model.sample_latent(
            x, num_samples=num_steps + 1, return_logits=True
        )
        h_old = self._init_h_old(idx, h_new_all[:, 0:1, ...])
        h_new_all[:, 0, ...] = h_old[:, 0, ...]

        log_p_all = self.joint_model.log_joint_prob_multiple_samples(x, h_new_all)
        log_q_all = self.proposal_model.log_prob_from_logits(h_new_all, h_logits)

        cur_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        traj_idxes = [] if return_all else None

        for step in range(num_steps):
            prob_idx = torch.full_like(cur_idx, step + 1)
            log_r = (
                log_p_all.gather(1, prob_idx.unsqueeze(1)).squeeze(1)
                - log_p_all.gather(1, cur_idx.unsqueeze(1)).squeeze(1)
                + log_q_all.gather(1, cur_idx.unsqueeze(1)).squeeze(1)
                - log_q_all.gather(1, prob_idx.unsqueeze(1)).squeeze(1)
            )
            accept = torch.rand_like(log_r) < torch.exp(torch.clamp(log_r, max=0.0, min=-100.0))
            self._accept_cnt += accept.sum()
            self._total_cnt += accept.numel()
            cur_idx = torch.where(accept, prob_idx, cur_idx)
            if return_all:
                traj_idxes.append(cur_idx.clone())

        if return_all:
            traj_idxes = torch.stack(traj_idxes, dim=1)
            h_result = h_new_all.gather(
                1,
                traj_idxes.view(
                    x.shape[0], num_steps, *([1] * (h_new_all.dim() - 2))
                ).expand(-1, -1, *h_new_all.shape[2:]),
            )
            h_final = h_result[:, -1:, ...]
        else:
            h_result = h_new_all.gather(
                1,
                cur_idx.view(-1, 1, *([1] * (h_new_all.dim() - 2))).expand(
                    -1, 1, *h_new_all.shape[2:]
                ),
            )
            h_final = h_result

        if self.use_cache and idx is not None:
            self.cache[idx] = h_final.squeeze(1).detach().to(self.cache.dtype)
            self.updated_mask[idx] = True

        return h_result

    @torch.no_grad()
    def get_acceptance_rate(self):
        accept = self._accept_cnt.clone()
        total = self._total_cnt.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(accept, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if total.item() == 0:
            return 0.0
        return (accept / total).item()

    def state_dict(self):
        if not self.use_cache:
            return {}
        return {"cache": self.cache.cpu()}

    def load_state_dict(self, state):
        if "cache" in state:
            self.cache = state["cache"].to(next(self.proposal_model.parameters()).device)

    @torch.no_grad()
    def sync_cache(self):
        if not self.use_cache:
            return
        if not dist.is_available() or not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        orig_dtype = self.cache.dtype
        local_cache = self.cache.to(torch.int32) if orig_dtype in [torch.int8, torch.int16] else self.cache
        local_mask = self.updated_mask

        cache_list = [torch.empty_like(local_cache) for _ in range(world_size)]
        mask_list = [torch.empty_like(local_mask) for _ in range(world_size)]
        dist.all_gather(cache_list, local_cache)
        dist.all_gather(mask_list, local_mask)

        merged_cache = local_cache.clone()
        for rank in range(world_size):
            merged_cache[mask_list[rank]] = cache_list[rank][mask_list[rank]]

        self.cache.copy_(merged_cache.to(orig_dtype))
        self.updated_mask.zero_()


if __name__ == "__main__":
    pass
