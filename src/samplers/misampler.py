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
    ):
        super().__init__()
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.dataset_size = dataset_size
        self.use_cache = use_cache  # use the setter, which initializes cache if needed

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value: bool):
        self._use_cache = value
        if value and not hasattr(self, "cache"):
            # initialize cache if not present
            self._init_cache(self.dataset_size, self.proposal_model.num_latent_vars)

    @torch.no_grad()
    def _init_cache(self, dataset_size, num_latent_vars):
        """Initialize cache by sampling from proposal model

        You may modify this method to change the initialization strategy.
        """
        self.cache = torch.full(
            (dataset_size, num_latent_vars),
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
        h_new = h_new.squeeze(-1).long()  # [batch_size, num_latent_vars]
        if self.use_cache and idx is not None:
            h_old = self.cache[idx]  # [batch_size, num_latent_vars], dtype=torch.long
            uninitialized_mask = h_old == -1
            if uninitialized_mask.any():
                h_old[uninitialized_mask] = h_new[uninitialized_mask].clone()
        else:
            h_old = h_new.clone()
        return h_old.unsqueeze(-1).float()  # [batch_size, num_latent_vars, 1]

    @torch.no_grad()
    def _cal_acceptance_prob(self, x, h_new, h_old):
        """
        Calculate acceptance probability for MIS step
        a = p(x,h') * q(h|x) / (p(x,h) * q(h'|x))
        """
        log_p_new = self.joint_model.log_joint_prob(x, h_new)
        log_p_old = self.joint_model.log_joint_prob(x, h_old)

        log_q_new = self.proposal_model.log_conditional_prob(h_new, x)
        log_q_old = self.proposal_model.log_conditional_prob(h_old, x)

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

        log_accept = (log_p_new + log_q_old) - (log_p_old + log_q_new)
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
        # h_old: [batch_size, num_latent_vars, 1] or None
        # idx: [batch_size, ]

        # Get h_old from cache if using cache
        if h_old is None:
            # If there is no h_old provided, initialize from cache or proposal model
            h_old = self._init_h_old(idx, self.proposal_model.sample_latent(x))

        # Propose from q_phi(h|x)
        h_new = self.proposal_model.sample_latent(x)  # [batch_size, num_latent_vars, 1]

        # Compute acceptance probability
        accept_prob = self._cal_acceptance_prob(x, h_new, h_old)  # [batch_size, 1]

        u = torch.rand_like(accept_prob)
        accept = (u < accept_prob).float().unsqueeze(-1)  # [batch_size, 1, 1]

        # Update sample
        h_next = (
            accept * h_new + (1 - accept) * h_old
        )  # [batch_size, num_latent_vars, 1]
        h_next = h_next.round()  # Avoid numerical issues

        # update cache
        if self.use_cache and idx is not None:
            self.cache[idx] = h_next.detach().squeeze(-1).long()
            self.updated_mask[idx] = True  # mark as updated

        return h_next

    @torch.no_grad()
    def sample(self, x, idx=None, num_steps=1, parallel=True):
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

        # Initialize h_old from cache or proposal model
        h_old = self._init_h_old(
            idx, self.proposal_model.sample_latent(x)
        )  # [batch_size, num_latent_vars, 1]

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
