# src/samplers/misampler.py
import torch
from src.base.base_sampler import BaseSampler
import torch.distributed as dist


class MISampler(BaseSampler):
    """Metropolis Independence Sampler (MIS) for sampling latent variables h ~ p_theta (h|x)

    Algorithm:
        1. Propose h' ~ q_phi(h|x)
        2. Compute acceptance probability:
            a = p_theta(x,h') * q_phi(h|x) / (p_theta(x,h) * q_phi(h'|x))
        3. Accept/reject:
            h = h' with probability min(1, a)
            h = h  with probability 1 - min(1, a)

    Cache mechanism:
        - If use_cache=True, maintain a cache of latent variables for each data point.
        - During sampling, retrieve h_old from cache for the given data index.
        - After sampling, update the cache with the new accepted h.

    Args:
        joint_model: The joint model p_theta(x,h)
        proposal_model: The proposal model q_phi(h|x)
        use_cache: Whether to use cache mechanism
        dataset_size: Size of the dataset (required if use_cache is True)
    """

    def __init__(
        self,
        joint_model,  # p(x, h)
        proposal_model,  # q(h|x)
        use_cache=False,  # whether to use cache mechanism
        dataset_size=None,  # required if use_cache is True
        element_shape=None,  # for picture, is (H, W)
    ):
        super().__init__()
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.dataset_size = dataset_size
        self.use_cache = use_cache  # use the setter, which initializes cache if needed
        self.element_shape = element_shape

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value: bool):
        self._use_cache = value
        if value and not hasattr(self, "cache"):
            # initialize cache if not present
            self._init_cache(
                self.dataset_size,
                self.proposal_model.num_latent_vars,
                self.element_shape,
            )

    @torch.no_grad()
    def _init_cache(self, dataset_size, num_latent_vars, element_shape):
        """Initialize cache by sampling from proposal model

        You may modify this method to change the initialization strategy.
        """
        cache_shape = (
            (dataset_size, *element_shape, num_latent_vars)
            if element_shape is not None
            else (dataset_size, num_latent_vars)
        )  # shape of the cache, for picture data, is (N, H, W, num_latent_vars)
        self.cache = torch.full(
            cache_shape,
            int(-1),
            dtype=torch.long,
            device=next(self.proposal_model.parameters()).device,
        )

        self.updated_mask = torch.zeros(
            dataset_size,
            dtype=torch.bool,
            device=next(self.proposal_model.parameters()).device,
        )

    def to(self, device):
        """Move sampler to device, including cache if present"""
        if self.use_cache:
            self.cache = self.cache.to(device)
            self.updated_mask = self.updated_mask.to(device)
        return self

    @torch.no_grad()
    def _init_h_old(self, idx, h_new):
        # h_new: [B, num_samples, ..., num_latent_vars]
        if self.use_cache and idx is not None:
            h_old = self.cache[
                idx
            ]  # [batch_size, H, W, num_latent_vars], dtype=torch.long
            h_old = (
                h_old.unsqueeze(1).expand_as(h_new).float()
            )  # [batch_size, num_samples, H, W, num_latent_vars]
            uninitialized_mask = (h_old[:, 0:1, ...] == -1).all(
                dim=tuple(range(2, h_old.dim()))
            )  # [batch_size, 1], check if all latent vars are -1
            if uninitialized_mask.any():
                mask_indices = torch.nonzero(
                    uninitialized_mask.squeeze(1), as_tuple=True
                )[
                    0
                ]  # get indices of uninitialized samples
                if len(mask_indices) > 0:
                    h_old[mask_indices] = h_new[mask_indices].clone()
        else:
            h_old = h_new.clone()
        return h_old  # [B, num_samples, ..., num_latent_vars]

    @torch.no_grad()
    def _cal_acceptance_prob(self, x, h_new, h_old):
        """Calculate acceptance probability for MIS step

        a = p(x,h') * q(h|x) / (p(x,h) * q(h'|x))
        """
        
        # x: [B, ...]
        # h_new, h_old: [B, num_samples, ..., num_latent_vars]
        batch_size, num_samples = h_new.shape[0], h_new.shape[1]
        h_new_flat = h_new.reshape(batch_size * num_samples, *h_new.shape[2:])  # [B * num_samples, ...]
        h_old_flat = h_old.reshape(batch_size * num_samples, *h_old.shape[2:])  # [B * num_samples, ...]

        log_p_new = self.joint_model.log_joint_prob(x, h_new_flat).reshape(batch_size, num_samples) # [B,num_samples]
        log_p_old = self.joint_model.log_joint_prob(x, h_old_flat).reshape(batch_size, num_samples) # [B,num_samples]

        log_q_new = self.proposal_model.log_conditional_prob(h_new, x) # [B, num_samples]
        log_q_old = self.proposal_model.log_conditional_prob(h_old, x) # [B, num_samples]

        # Check for NaN or Inf
        if (
            torch.isnan(log_p_new).any()
            or torch.isnan(log_p_old).any()
            or torch.isnan(log_q_new).any()
            or torch.isnan(log_q_old).any()
        ):
            raise ValueError(
                "NaN encountered in log probabilities during acceptance probability calculation."
            )
        if (
            torch.isinf(log_p_new).any()
            or torch.isinf(log_p_old).any()
            or torch.isinf(log_q_new).any()
            or torch.isinf(log_q_old).any()
        ):
            raise ValueError(
                "Infinity encountered in log probabilities during acceptance probability calculation."
            )

        log_accept = (log_p_new + log_q_old) - (log_p_old + log_q_new) # [B, num_samples]
        accept_prob = torch.exp(torch.clamp(log_accept, max=0.0, min=-100.0)) # [B, num_samples]
        return accept_prob # [B, num_samples]

    @torch.no_grad()
    def _cal_acceptance_prob_faster(self, x, h_new, h_old):
        """Calculate acceptance probability for MIS step. Only for categorical latent variables and
        gaussian joint model.

        log a = log p(x,h') + log q(h|x) - log p(x,h) - log q(h'|x)

        log p(x,h') - log p(x,h) = log N(x; mu(h'), sigma) - log N(x; mu(h), sigma)
        = 1/sigma^2 (mu(h') - mu(h))^T (x - 0.5 (mu(h') + mu(h)))

        log q(h|x) and log q(h'|x) can be computed: (Only for one codebook categorical latent variables)
        log q(h|x) = logit_h - logsumexp_logits
        log q(h'|x) = logit_h' - logsumexp_logits
        so, log q(h|x) - log q(h'|x) = logit_h - logit_h'

        """
        raise NotImplementedError("This faster acceptance probability calculation is not implemented yet.")
        assert (
            self.joint_model.__class__.__name__ == "JointModelCategoricalGaussian"
            and self.proposal_model.__class__.__name__ == "ProposalModelCategorical"
            and self.proposal_model.num_categories is not None
            and len(self.proposal_model.num_categories) == 1
        ), "This faster acceptance probability calculation only supports categorical latent variables with one codebook and gaussian joint model."

        mu_h_new = self.joint_model(h_new)  # [B, input_dim, num_samples]
        mu_h_old = self.joint_model(h_old)  # [B, input_dim, num_samples]

        sigma_sq = self.joint_model.sigma**2  # scalar

        x_expand = x.unsqueeze(-1)  # [B, input_dim, 1]
        term1 = (mu_h_new - mu_h_old).permute(0, 2, 1)  # [B, num_samples, input_dim]
        term2 = x_expand - 0.5 * (mu_h_new + mu_h_old)  # [B, input_dim, num_samples]
        log_p_diff = (1.0 / sigma_sq) * torch.bmm(term1, term2).squeeze(
            -1
        )  # [B, num_samples]
        logits = self.proposal_model.forward(x)[0]  # [B, latent_dim]
        # get logits for h_new and h_old
        logits_h_new = torch.gather(
            logits, 1, h_new.squeeze(-1).long()
        )  # [B, num_samples]
        logits_h_old = torch.gather(
            logits, 1, h_old.squeeze(-1).long()
        )  # [B, num_samples]

        log_q_diff = logits_h_old - logits_h_new  # [B, num_samples]

        log_accept = log_p_diff + log_q_diff  # [B, num_samples]
        accept_prob = torch.exp(torch.clamp(log_accept, max=0.0, min=-100.0))
        return accept_prob

    @torch.no_grad()
    def step(self, x, idx=None, h_old=None):
        """
        Perform single MIS step:
            propose h'
            compute acceptance probability
            accept/reject
        """

        # x: [batch_size, ...]
        # h_old: [B, num_samples, ..., num_latent_vars] or None
        # idx: [batch_size, ]

        # Get h_old from cache if using cache
        if h_old is None:
            # If there is no h_old provided, initialize from cache or proposal model
            h_old = self._init_h_old(idx, self.proposal_model.sample_latent(x))

        # Propose from q_phi(h|x)
        h_new = self.proposal_model.sample_latent(x)  # [batch_size, num_samples, ..., num_latent_vars]

        # Compute acceptance probability
        accept_prob = self._cal_acceptance_prob(x, h_new, h_old)  # [batch_size, num_samples]
        # accept_prob = self._cal_acceptance_prob_faster(
        #     x, h_new, h_old
        # )  # [batch_size, num_samples]

        u = torch.rand_like(accept_prob)
        accept = (u < accept_prob).float()  # [batch_size, num_samples]

        target_shape = accept.shape + (1,) * (h_new.dim() - accept.dim())
        accept = accept.view(target_shape)  # reshape for broadcasting, [batch_size, num_samples, 1, ..., 1]
        # Update sample
        h_next = (
            accept * h_new + (1 - accept) * h_old
        )  # [batch_size, num_samples, ..., num_latent_vars]
        h_next = h_next.round()  # Avoid numerical issues

        # update cache
        if self.use_cache and idx is not None:
            # Randomly pick one sample to update the cache
            rand_indices = torch.randint(0, h_next.shape[1], (h_next.shape[0],))
            h_next_selected = h_next[
                torch.arange(h_next.shape[0]), rand_indices
            ]  # [batch_size, ..., num_latent_vars]
            self.cache[idx] = h_next_selected.detach().long()
            self.updated_mask[idx] = True  # mark as updated

        return h_next

    @torch.no_grad()
    def sample(self, x, idx=None, num_steps=1, parallel=False):
        """Generate samples using MIS sampler.


        Sequential multi-step sampling for MIS.

        """

        # if not self.use_cache and num_steps > 1:
        #     raise ValueError(
        #         "Multi-step sampling (num_steps > 1) is not meaningful when use_cache=False."
        #     )

        if parallel:
            return self._sample_parallel(x, idx=idx, num_steps=num_steps)
        else:
            h_old = None
            for _ in range(num_steps):
                h_old = self.step(x, idx=idx, h_old=h_old)

        return h_old

    @torch.no_grad()
    def _sample_parallel(self, x, idx=None, num_steps=1):
        """Parallelized multi-step sampling for MIS. Generativetes all proposal samples in parallel.

        Args:
            x: Input data, shape [batch_size, ...].
            idx: Indices for cache, shape [batch_size].
            num_steps: Number of sampling steps.
        Returns:
            h_final: Final sampled latent variables, shape [batch_size, latent_dim].
        """

        # Check if multi-step sampling is meaningful
        # if not self.use_cache and num_steps > 1:
        #     raise ValueError(
        #         "Multi-step sampling (num_steps > 1) is not meaningful when use_cache=False."
        #     )

        raise NotImplementedError("This parallel multi-step sampling is not implemented yet.")
        # Initialize h_old from cache or proposal model
        h_old = self._init_h_old(
            idx,
            self.proposal_model.sample_latent(
                x
            ),  # [B, num_samples, ..., num_latent_vars]
        )

        # Generate all proposal samples in parallel
        h_new = self.proposal_model.sample_latent(
            x, num_samples=num_steps
        )  # [Batch, num_latent_vars, num_steps]

        # Sequentially compute acceptance probabilities and update samples
        for t in range(num_steps):
            # Get the t-th proposal sample
            h_new_t = (
                h_new[:, :, t].unsqueeze(-1).float()
            )  # [batch_size, num_latent_vars, 1]

            # Compute acceptance probability
            accept_prob = self._cal_acceptance_prob(x, h_new_t, h_old)

            # Accept/reject step
            u = torch.rand_like(accept_prob)
            accept = (u < accept_prob).float().unsqueeze(-1)  # [batch_size, 1, 1]

            # Update samples
            h_old = accept * h_new_t + (1 - accept) * h_old
            h_old = h_old.round()  # Avoid numerical issues

        # Update cache
        if self.use_cache and idx is not None:
            self.cache[idx] = h_old.detach().squeeze(-1).long()
            self.updated_mask[idx] = True  # mark as updated

        return h_old

    def state_dict(self):
        if not self.use_cache:
            return {}

        return {
            "cache": self.cache.cpu(),
        }

    def load_state_dict(self, state):
        if "cache" in state:
            self.cache = state["cache"].to(
                next(self.proposal_model.parameters()).device
            )

    @torch.no_grad()
    def sync_cache(self):
        """
        Synchronize cache across all ranks.



        Theoretically, our h is discrete, so averaging may not be ideal or even correct. However,
        in practice, this works reasonably well because there will not two ranks update the same cache entry.
        DDP ensures each rank has different data samples, so the cache indices updated by different ranks
        should be different. Thus, averaging is equivalent to taking the non-nan value.

        """
        if not self.use_cache:
            return  # No cache to sync

        if not dist.is_available() or not dist.is_initialized():
            return  # No need to sync if not distributed

        world_size = dist.get_world_size()
        local_cache = self.cache
        local_mask = self.updated_mask

        cache_list = [torch.empty_like(local_cache) for _ in range(world_size)]
        mask_list = [torch.empty_like(local_mask) for _ in range(world_size)]

        dist.all_gather(cache_list, local_cache)
        dist.all_gather(mask_list, local_mask)

        merged_cache = local_cache.clone()

        for r in range(world_size):
            mask_r = mask_list[r]
            cache_r = cache_list[r]
            merged_cache[mask_r] = cache_r[mask_r]

        self.cache.copy_(merged_cache)
        self.updated_mask.zero_()  # reset updated mask after sync


if __name__ == "__main__":

    # Test code need to be after the joint model and proposal model are implemented
    pass
