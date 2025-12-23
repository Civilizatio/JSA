# src/models/components/proposal_model.py
import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseProposalModel


class ProposalModelBernoulli(BaseProposalModel):
    """q_phi(h|x)

    Must implement functions:
    - log_conditional_prob(h, x)
    - sample_latent(x)

    We assume Bernoulli distribution for q_phi(h|x)
    """

    def __init__(
        self,
        net: nn.Module = None,
        num_latent_vars=256,
    ):
        super().__init__()

        self.num_latent_vars = num_latent_vars
        self._latent_dim = num_latent_vars

        self._categories = [2] * num_latent_vars  # for compatibility
        self.net = net

    @property
    def latent_dim(self):
        return self._latent_dim

    def forward(self, x):
        """
        Compute distribution parameters (logits) for q(h|x).

        Args:
            x: [B, input_dim]

        Returns:
            logits: [B, num_latent_vars]
        """
        logits = self.net(x)  # [B, latent_dim]
        return logits

    def sample_latent(self, x, num_samples=1):
        """Sample h ~ q(h|x)"""
        logits = self.forward(x)  # [B, latent_dim]
        probs = torch.sigmoid(logits)  # [B, latent_dim]

        # Expand for num_samples
        probs_expanded = probs.unsqueeze(-1).expand(
            -1, -1, num_samples
        )  # [B, latent_dim, num_samples]

        # Bernoulli sampling requires input shape [B, latent_dim, num_samples]
        # But torch.bernoulli expects the last dim to be the one we sample from usually,
        # actually it samples element-wise.
        h_samples = torch.bernoulli(probs_expanded)  # [B, latent_dim, num_samples]

        return h_samples

    def encode(self, x):
        """Deterministic encoding (mode of the distribution)"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        h = (probs > 0.5).float()
        return h  # [B, latent_dim]

    def log_conditional_prob(self, h, x):
        """
        Compute log q(h|x)

        Args:
            h: [B, latent_dim] or [B, latent_dim, num_samples]
            x: [B, input_dim]

        Returns:
            log_prob: [B, num_samples] (summed over latent_dim)
        """
        logits = self.forward(x)  # [B, latent_dim]

        # Handle dimensions
        if h.dim() == 2:
            h = h.unsqueeze(-1)  # [B, latent_dim, 1]

        # logits: [B, latent_dim] -> [B, latent_dim, 1]
        logits = logits.unsqueeze(-1)

        # BCE with logits = -log q(h|x)
        # We want log q(h|x), so we negate BCE
        # h must be broadcastable to logits
        log_prob_elementwise = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits.expand_as(h), h, reduction="none"
        )  # [B, latent_dim, num_samples]

        return log_prob_elementwise.sum(dim=1)  # [B, num_samples]

    def get_loss(self, h, x):
        """
        Compute negative log conditional probability as loss

        Args:
            x: observed data
            h: latent variables (targets)

        Returns:
            loss: scalar
        """
        log_cond = self.log_conditional_prob(h, x)  # [B, num_samples]
        return -log_cond.mean()


class ProposalModelCategorical(BaseProposalModel):
    """q_phi(h|x)

    Must implement functions:
    - log_conditional_prob(h, x)
    - sample_latent(x)

    We assume Categorical distribution for q_phi(h|x)
    """

    def __init__(
        self,
        net: nn.Module = None,
        num_latent_vars=10,
        num_categories=256,
    ):
        super().__init__()

        self.num_latent_vars = num_latent_vars

        if isinstance(num_categories, int):
            self._num_categories = [num_categories] * num_latent_vars
        elif len(num_categories) == 1 and num_latent_vars > 1:
            self._num_categories = list(num_categories) * num_latent_vars
        else:
            assert (
                len(num_categories) == num_latent_vars
            ), "num_categories must be an integer or a list of length num_latent_vars"
            self._num_categories = list(num_categories)

        self.total_num_categories = sum(self._num_categories)
        self.net = (
            net  # nn.Module that outputs logits of shape [B, ..., total_num_categories]
        )

    @property
    def latent_dim(self):
        return self.total_num_categories

    def forward(self, x):
        """Compute distribution parameters (logits) for q(h|x).

        Args:
            x: [B, ...]

        Returns:
            split_logits: List of tensors, whose length is self.num_latent_vars,
                each shape [B, num_categories_i]
        """
        logits = self.net(
            x
        )  # [B, total_num_categories] or [B, ..., total_num_categories]
        split_logits = torch.split(
            logits, self._num_categories, dim=-1
        )  # List of [B, ..., num_categories_i]
        return split_logits

    def sample_latent(self, x, num_samples=1, return_logits=False):
        """Sample h ~ q(h|x)

        Returns:
            h_samples: Tensor of shape [B, num_samples, ..., num_latent_vars],
                containing sampled latent variable indices. dtype=torch.float
        """
        split_logits = self.forward(x)  # List of [B, ..., num_categories_i]

        h_samples_list = []
        for _, logits in enumerate(split_logits):
            # logits: [B, ..., num_categories_i]
            dist = torch.distributions.Categorical(logits=logits)
            h_samples = dist.sample([num_samples])  # [num_samples, B, ...]
            h_samples = h_samples.permute(
                1, 0, *range(2, h_samples.dim())
            ).contiguous()  # [B, num_samples, ...]
            h_samples.unsqueeze_(-1)  # [B, num_samples, ..., 1]
            h_samples_list.append(h_samples)

        h_samples = torch.cat(
            h_samples_list, dim=-1
        )  # [B, num_samples, ..., num_latent_vars]
        
        if return_logits:
            return h_samples.float(), split_logits  # dtype=torch.float
        return h_samples.float()  # dtype=torch.float

    def encode(self, x):
        """Deterministic encoding (argmax)"""
        split_logits = self.forward(x)

        idx_list = []
        for logits in split_logits:
            # logits: [B, ..., num_categories_i]
            idx = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
            idx_list.append(idx)

        h_idx = torch.cat(idx_list, dim=-1)  # [B, ..., num_latent_vars]
        return h_idx.float()

    def log_conditional_prob(self, h, x):
        """Compute log q(h|x)

        Args:
            h: Tensor of latent variable indices, shape [B, ..., num_latent_vars], or [B, num_samples, ..., num_latent_vars]
            x: Input tensor of shape [B, ...].

        Attention:
            `...` in h and x are not the same, they represent different dimensions.
            For example, if x has shape [B, C, H, W], then ... in x represents [C, H, W].
            However, h has shape [B, H, W, num_latent_vars], then ... in h represents [H, W].
        Returns:
            log_cond: Tensor of shape [B] containing log probabilities log q(h|x).
        """
        split_logits = self.forward(x)  # List of [B, ..., num_categories_i]
        log_cond = self.log_prob_from_logits(h, split_logits)  # [B, num_samples]
        return log_cond  # [B, num_samples]
    
    def log_prob_from_logits(self, h, split_logits):
        """ Compute log q(h|x) given precomputed split logits/
        
        Args:
            h: Tensor of latent variable indices, shape [B, ..., num_latent_vars], or [B, num_samples, ..., num_latent_vars]
            split_logits: List of tensors, whose length is self.num_latent_vars,
                each shape [B, ..., num_categories_i]
        Returns:
            log_cond: Tensor of shape [B, num_samples] containing log probabilities log q(h|x).
        """
        if h.dim() == split_logits[0].dim():
            h = h.unsqueeze(1)  # [B, 1, ..., num_latent_vars]

        log_cond = 0.0
        for i, logits in enumerate(split_logits):
            # logits: [B, ..., num_categories_i]
            logits_expanded = logits.unsqueeze(1)  # [B, 1, ..., num_categories_i]
            dist = torch.distributions.Categorical(logits=logits_expanded)

            h_i = h[..., i]  # [B, num_samples, ...]

            log_cond_i = dist.log_prob(h_i.long()).sum(
                dim=tuple(range(2, h_i.dim()))
            )  # [B, num_samples]
            log_cond += log_cond_i

        return log_cond  # [B, num_samples]
    
    def log_conditional_prob_diff(self, x, h_new, h_old, split_logits=None):
        """
        Calculate log q(h_new|x) - log q(h_old|x) efficiently.
        
        Args:
            x: [B, ...]
            h_new: [B, num_samples, ..., num_latent_vars]
            h_old: [B, num_samples, ..., num_latent_vars]
            split_logits: Optional precomputed logits. List of [B, ..., num_categories]
            
        Returns:
            log_q_diff: [B, num_samples]
        """
        if split_logits is None:
            split_logits = self.forward(x)
            
        if isinstance(split_logits, torch.Tensor):
            split_logits = [split_logits]
            
        log_q_diff = 0.0
        
        for i, logits in enumerate(split_logits):
            # logits: [B, ..., num_categories_i]
            logits_expanded = logits.unsqueeze(1) # [B, 1, ..., num_categories_i]
            
            h_new_i = h_new[..., i].long() # [B, num_samples, ...]
            h_old_i = h_old[..., i].long() # [B, num_samples, ...]
            
            # Gather logits
            # logits_expanded: [B, 1, ..., C]
            # h indices: [B, S, ...] -> [B, S, ..., 1]
            
            logits_h_new = torch.gather(logits_expanded, -1, h_new_i.unsqueeze(-1)).squeeze(-1)
            logits_h_old = torch.gather(logits_expanded, -1, h_old_i.unsqueeze(-1)).squeeze(-1)
            
            # Sum over spatial dimensions if any (dims between 1 (samples) and last)
            if logits_h_new.dim() > 2:
                dims_to_sum = tuple(range(2, logits_h_new.dim()))
                logits_h_new = logits_h_new.sum(dim=dims_to_sum)
                logits_h_old = logits_h_old.sum(dim=dims_to_sum)
                
            log_q_diff += (logits_h_new - logits_h_old)
            
        return log_q_diff # [B, num_samples]

    def get_loss(self, h, x):
        """Compute negative log conditional probability as loss"""
        log_cond = self.log_conditional_prob(h, x)  # [B, num_samples]
        return -log_cond.mean()  # scalar
