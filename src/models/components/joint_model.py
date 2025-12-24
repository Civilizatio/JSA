# src/models/components/joint_model.py

import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseJointModel
import math


class JointModelBernoulliBernoulli(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)

    We assume Bernoulli prior for p(h) and Bernoulli likelihood for p(x|h)
    """

    def __init__(
        self,
        net: nn.Module,
        num_latent_vars=256,
    ):
        super().__init__()

        self._latent_dim = num_latent_vars
        self.num_latent_vars = num_latent_vars

        self.net = net

    @property
    def latent_dim(self):
        return self._latent_dim

    def log_prior_prob(self, h):
        """Compute log p(h)"""
        # For Bernoulli latent variables with uniform prior
        return h.size(1) * torch.log(torch.tensor(0.5, device=h.device))

    def forward(self, h):
        """Compute p(x|h) parameters (probabilities)

        Args:
            h: latent variables, shape [B, num_latent_vars] or [B, num_latent_vars, num_samples]

        Returns:
            probs_x: shape [B, output_dim, num_samples]
        """
        if h.dim() == 2:  # [B, num_latent_vars]
            h = h.unsqueeze(-1)  # [B, num_latent_vars, 1]

        B, num_latent_vars, num_samples = h.size()

        # Reshape h for MLP: [B, num_latent_vars, num_samples] -> [B, num_samples, num_latent_vars]
        h_reshaped = h.transpose(1, 2).reshape(
            -1, num_latent_vars
        )  # [B * num_samples, num_latent_vars]

        # Assume net includes Sigmoid activation and returns probabilities
        probs_x = self.net(h_reshaped)  # [B * num_samples, output_dim]

        probs_x = probs_x.view(B, num_samples, -1)  # [B, num_samples, output_dim]

        return probs_x.transpose(1, 2)  # [B, output_dim, num_samples]

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)"""
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        probs_x = self.forward(h)  # [B, output_dim, num_samples]
        x = x.unsqueeze(-1)  # [B, output_dim, 1]

        # Bernoulli Log Prob
        # We use binary_cross_entropy (which is -log_prob) so we negate it
        # Note: x must be broadcastable to probs_x
        log_p_x_given_h = -nn.functional.binary_cross_entropy(
            probs_x, x.expand_as(probs_x), reduction="none"
        ).sum(
            dim=1
        )  # Sum over output_dim, shape [B, num_samples]

        return log_p_h + log_p_x_given_h

    def get_loss(self, x, h):
        """Compute negative log joint probability as loss (ignoring prior in loss calculation as per template)"""
        probs_x = self.forward(h)  # [B, output_dim, num_samples]
        x = x.unsqueeze(-1)  # [B, output_dim, 1]

        # Using BCE loss as negative log likelihood
        log_p_x_given_h = -nn.functional.binary_cross_entropy(
            probs_x, x.expand_as(probs_x), reduction="none"
        ).sum(
            dim=1
        )  # shape [B, num_samples]

        log_joint = log_p_x_given_h.mean(dim=1)  # average over samples, shape [B]
        return -log_joint.mean()

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)"""
        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            )  # shape [1, num_latent_vars]
            # Expand h to generate num_samples if needed, though forward handles expansion
            # if h is [1, num_latent_vars], forward returns [1, output_dim, 1]
            # To get num_samples distinct samples from one h, we rely on bernoulli sampling below

        probs_x = self.forward(
            h
        )  # [B, output_dim, num_samples] (if h had samples) or [B, output_dim, 1]

        if num_samples > 1 and probs_x.shape[2] == 1:
            probs_x = probs_x.expand(-1, -1, num_samples)

        x_samples = torch.bernoulli(probs_x)  # [B, output_dim, num_samples]
        return x_samples

    def decode(self, h):
        """Decode x ~ p(x|h)"""
        probs_x = self.forward(h).squeeze(-1)  # [B, output_dim]
        x = (probs_x > 0.5).float()
        return x


class JointModelBernoulliGaussian(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Bernoulli prior for p(h) and Gaussian likelihood for p(x|h).
    """

    SIGMA = 0.1  # Fixed standard deviation for Gaussian likelihood

    def __init__(
        self,
        net: nn.Module,
        num_latent_vars=256,
    ):
        super().__init__()

        self._latent_dim = num_latent_vars
        self.num_latent_vars = num_latent_vars
        self.net = net

    @property
    def latent_dim(self):
        return self._latent_dim

    def log_prior_prob(self, h):
        """Compute log p(h)"""
        return h.size(1) * torch.log(torch.tensor(0.5, device=h.device))

    def forward(self, h):
        """Compute p(x|h) parameters (mean)

        Args:
            h: latent variables, shape [B, num_latent_vars] or [B, num_latent_vars, num_samples]

        Returns:
            mean_x: shape [B, output_dim, num_samples]
        """
        if h.dim() == 2:  # [B, num_latent_vars]
            h = h.unsqueeze(-1)  # [B, num_latent_vars, 1]

        B, num_latent_vars, num_samples = h.size()

        # Reshape h: [B, num_latent_vars, num_samples] -> [B, num_samples, num_latent_vars]
        h_reshaped = h.transpose(1, 2).reshape(
            -1, num_latent_vars
        )  # [B * num_samples, num_latent_vars]

        # Assume net includes appropriate activation (e.g. Sigmoid) and returns mean in [0, 1]
        mean_x = self.net(h_reshaped)  # [B * num_samples, output_dim]

        mean_x = mean_x.view(B, num_samples, -1)  # [B, num_samples, output_dim]

        return mean_x.transpose(1, 2)  # [B, output_dim, num_samples]

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)"""
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        mean_x = self.forward(h)  # [B, output_dim, num_samples]
        x = x.unsqueeze(-1)  # [B, output_dim, 1]

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        log_p_x_given_h = gaussian_dist.log_prob(x).sum(
            dim=1
        )  # Sum over output_dim, shape [B, num_samples]

        return log_p_h + log_p_x_given_h

    def get_loss(self, x, h):
        """Compute negative log joint probability as loss"""
        mean_x = self.forward(h)  # [B, output_dim, num_samples]
        x = x.unsqueeze(-1)  # [B, output_dim, 1]

        # Using MSE loss as negative log likelihood
        mse_loss = nn.MSELoss(reduction="none")
        log_p_x_given_h = -mse_loss(mean_x, x).sum(dim=1)  # shape [B, num_samples]

        log_joint = log_p_x_given_h.mean(dim=1)  # average over samples, shape [B]
        return -log_joint.mean()

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)"""
        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            )  # shape [1, num_latent_vars]

        mean_x = self.forward(h)  # [B, output_dim, 1] (if h has no samples)

        if num_samples > 1 and mean_x.shape[2] == 1:
            mean_x = mean_x.expand(-1, -1, num_samples)

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        x_samples = gaussian_dist.sample()  # [B, output_dim, num_samples]
        return torch.clamp(x_samples, 0.0, 1.0)

    def decode(self, h):
        """Decode x ~ p(x|h)"""
        mean_x = self.forward(h).squeeze(-1)  # [B, output_dim]
        return mean_x  # return mean as decoded output


class JointModelCategoricalGaussian(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Categorical prior for p(h) and Gaussian likelihood for p(x|h).
    """

    def __init__(
        self,
        net: nn.Module,
        num_categories,
        num_latent_vars,
        embedding_dims=None,
        sigma=0.1,
    ):
        super().__init__()

        self.num_latent_vars = num_latent_vars
        self.register_buffer("sigma", torch.tensor(float(sigma)))

        if len(num_categories) == 1 and num_latent_vars > 1:
            self._num_categories = list(num_categories) * num_latent_vars
        else:
            assert (
                len(num_categories) == num_latent_vars
            ), "num_categories must be an integer or a list of length num_latent_vars"
            self._num_categories = list(num_categories)

        # Add Embedding layer for Categorical latent variables
        if embedding_dims is None:
            self.embedding_dims = [
                min(2, int(math.log2(K))) for K in self._num_categories
            ]
        elif len(embedding_dims) == 1 and num_latent_vars > 1:
            self.embedding_dims = list(embedding_dims) * num_latent_vars
        else:
            assert (
                len(embedding_dims) == num_latent_vars
            ), "embedding_dims must be an integer or a list of length num_latent_vars"
            self.embedding_dims = list(embedding_dims)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=K, embedding_dim=emb_dim)
                for K, emb_dim in zip(self._num_categories, self.embedding_dims)
            ]
        )

        self._latent_dim = sum(self.embedding_dims)
        self.net = net

    @property
    def latent_dim(self):
        return self._latent_dim

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def log_prior_prob(self, h):
        """
        Compute log p(h)
        Assuming uniform prior over Categorical latent variables
        Mathmatically, for each latent variable with K categories, log p(h_i) = -log(K)
        Thus, log p(h) = sum over latent variables of -log(K_i)
        """
        num_categories_tensor = torch.tensor(
            self._num_categories, device=h.device, dtype=torch.float
        )
        log_p_h = -torch.sum(torch.log(num_categories_tensor))
        return log_p_h

    def log_joint_prob_multiple_samples(self, x, h):
        """Compute log p(x, h) for multiple samples of h

        Args:
            x: observed data, shape [B, ...]
            h: latent variables, shape [B, num_samples, ..., num_latent_vars]
        Returns:
            log_probs: shape [B, num_samples]
        """
        batch_size, num_samples = h.shape[0], h.shape[1]
        # Flatten first two dimensions to compute log_joint_prob
        h_flat = h.reshape(
            batch_size * num_samples, *h.shape[2:]
        )  # [B * num_samples, ..., num_latent_vars]
        log_probs_flat = self.log_joint_prob(
            x.repeat_interleave(num_samples, dim=0), h_flat
        )  # [B * num_samples]
        log_probs = log_probs_flat.view(batch_size, num_samples)  # [B, num_samples]
        return log_probs

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)

        Args:
            x: observed data, shape [B, ...]
            h: latent variables, shape [N, ..., num_latent_vars]
                if there are multiple samples, whose shape is [B, num_samples, ..., num_latent_vars], you
                should flatten the first two dimensions before passing in.
                So, N = B * num_samples in that case.

        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Gaussian distribution for each pixel with fixed variance (e.g., 1.0)
        # Shape of h: [B, ..., num_latent_vars] or [B, num_samples, ..., num_latent_vars]
        mean_x = self.forward(h)  # [B, ...]

        num_samples = 1
        if x.shape[0] != mean_x.shape[0]:  # multiple samples case
            num_samples = mean_x.shape[0] // x.shape[0]
            x = x.repeat_interleave(num_samples, dim=0)  # [N, ...]

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.sigma)
        log_p_x_given_h = gaussian_dist.log_prob(x).sum(
            dim=list(range(1, x.dim()))
        )  # sum over dimensions, shape [N,]

        log_p_x_given_h = log_p_x_given_h.view(-1, num_samples)  # [B, num_samples]
        return log_p_h + log_p_x_given_h  # log p(x, h)
    
    def log_joint_prob_diff(self, x, h_new, h_old):
        """Compute log p(x, h_new) - log p(x, h_old)

        Args:
            x: observed data, shape [B, ...]
            h_new: new latent variables, shape [B, num_samples, ..., num_latent_vars]
            h_old: old latent variables, shape [B, num_samples, ..., num_latent_vars]
            beta: to scale the difference, controls the sharpness of the distribution
        Returns:
            log_prob_diff: shape [B, num_samples]
        """
        batch_size, num_samples = h_new.shape[0], h_new.shape[1]
        
        h_new_flat = h_new.reshape(
            batch_size * num_samples, *h_new.shape[2:]
        )  # [B * num_samples, ..., num_latent_vars]
        
        h_old_flat = h_old.reshape(
            batch_size * num_samples, *h_old.shape[2:]
        )  # [B * num_samples, ..., num_latent_vars]
        
        mu_new = self.forward(h_new_flat)  # [B * num_samples, ...]
        mu_old = self.forward(h_old_flat)  # [B * num_samples, ...]
        
        mu_new = mu_new.view(batch_size, num_samples, *mu_new.shape[1:])  # [B, num_samples, ...]
        mu_old = mu_old.view(batch_size, num_samples, *mu_old.shape[1:])  # [B, num_samples, ...]
        
        x_expanded = x.unsqueeze(1)
        
        diff =  (mu_new-mu_old)*(x_expanded - 0.5*(mu_new + mu_old))/(self.sigma**2)
        log_prob_diff = diff.sum(dim=list(range(2, diff.dim())))  # sum over dimensions, shape [B, num_samples]
        
        return log_prob_diff  # shape [B, num_samples]

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)

        Args:
            h: latent variables, shape [B, ..., num_latent_vars]
            num_samples: number of samples to generate

        Returns:
            x_sample: sampled observed data, shape [B, num_samples, ...]
        """

        if h is None:
            # Sample h from uniform categorical distribution
            h_indices = [
                torch.randint(0, K, (1,)).to(next(self.parameters()).device)
                for K in self._num_categories
            ]
            h = torch.cat(h_indices, dim=-1)  # shape [1, num_latent_vars]

        mean_x = self.forward(h)  # [B, ...]
        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.sigma)
        x_samples = gaussian_dist.sample((num_samples,))  # [num_samples, B, ...]
        x_samples = (
            torch.clamp(x_samples, 0.0, 1.0)
            .permute(1, 0, *range(2, x_samples.dim()))
            .contiguous()
        )  # [B, num_samples, ...]
        return x_samples  # [B, num_samples, ...]

    def decode(self, h):
        """Decode x ~ p(x|h)

        Args:
            h: latent variables, shape [B, num_latent_vars]

        Returns:
            x_sample: sampled observed data, shape [B, output_dim]
        """

        mean_x = self.forward(h)  # [B, ...]
        return mean_x  # [B, output_dim]

    def get_loss(self, x, h, return_forward=False):
        """Compute negative log joint probability as loss

        Args:
            x: observed data, shape [B, ...]
            h: latent variables, shape [N, ..., num_latent_vars]
                if there are multiple samples, whose shape is [B, num_samples, ..., num_latent_vars], you
                should flatten the first two dimensions before passing in.
                So, N = B * num_samples in that case.
            return_forward: whether to return the forward output (mean_x)

        Returns:
            loss: negative log joint probability
            mean_x: decoded observed data, shape [N, ...] (only if return_forward is True)
        """

        mean_x = self.forward(h)  # [N, ...]
        if x.shape[0] != mean_x.shape[0]:  # multiple samples case
            x = x.repeat_interleave(mean_x.shape[0] // x.shape[0], dim=0)  # [N, ...]

        # Using MSE loss as negative log likelihood
        mse_loss = nn.MSELoss(reduction="mean")
        log_p_x_given_h = -mse_loss(mean_x, x)  # scalar
        log_joint = log_p_x_given_h
        if return_forward:
            return (
                -log_joint.mean(),
                mean_x,
            )  # negative log joint probability and forward output
        else:
            return -log_joint.mean()  # negative log joint probability

    def forward(self, h):
        """Decode x ~ p(x|h)

        Args:
            h: latent variables, shape [N, ..., num_latent_vars]

        Returns:
            mean_x: decoded observed data, shape [N, ...]
        """
        h_embedded = [
            embedding(h[..., i].long())  # [N, ...] -> [N, ..., embedding_dim_i]
            for i, embedding in enumerate(self.embeddings)
        ]
        h_densed = torch.cat(h_embedded, dim=-1)  # [N, ..., latent_dim]

        # Here, for MLP, h should be [N, latent_dim], we need to do nothing.
        # For CNN, h should be [N, H, W, latent_dim], we need to permute to [N, latent_dim, H, W].
        # However, we should not assume the net type here, so we just pass h_densed, and let the net handle it.

        mean_x = self.net(h_densed)  # [N, ...]
        return mean_x  # [N, ...]
