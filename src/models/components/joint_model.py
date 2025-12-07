# src/models/components/joint_model.py

import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseJointModel
from src.utils.mlp_utils import build_mlp
import math


class JointModelBernoulliBernoulli(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)

    We assume Bernoulli prior for p(h) and Bernoulli likelihood for p(x|h)
    Using MLP to model p_theta(x|h)
    """

    def __init__(
        self,
        num_latent_vars=256,
        layers=[512, 512],
        output_dim=784,
        activation: str = "relu",
        final_activation: str = None,
    ):
        super().__init__()

        self._latent_dim = num_latent_vars
        self.num_latent_vars = num_latent_vars

        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=num_latent_vars,
            layers=layers,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation,
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def log_prior_prob(self, h):
        """Compute log p(h)

        Args:
            h: latent variables, shape [B, num_latent_vars, num_samples]

        Returns:
            log p(h): shape []
        Assuming uniform prior over Bernoulli latent variables
        """
        # For Bernoulli latent variables with uniform prior,
        # log p(h) = num_latent_vars * log(0.5)
        return h.size(1) * torch.log(torch.tensor(0.5, device=h.device))  # shape []

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)

        Args:
            x: observed data, shape [B, output_dim]
            h: latent variables, shape [B, num_latent_vars, num_samples]

        Returns:
            log p(x, h): shape [B, num_samples]

        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)  # shape []

        # Likelihood p(x|h)
        # We assume independent Bernoulli distribution for each pixel
        B, num_latent_vars, num_samples = h.size()
        h_reshaped = h.transpose(1, 2).reshape(
            -1, num_latent_vars
        )  # [B * num_samples, num_latent_vars]
        logits_x = self.net(h_reshaped)  # [B * num_samples, output_dim]
        logits_x = logits_x.view(B, num_samples, -1)  # [B, num_samples, output_dim]
        x = x.unsqueeze(1)  # [B, 1, output_dim]
        log_p_x_given_h = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits_x, x, reduction="none"
        ).sum(
            dim=2
        )  # sum over dimensions, shape [B, num_samples]
        return log_p_h + log_p_x_given_h  # log p(x, h), shape [B, num_samples]

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)

        Args:
            h: latent variables, shape [B, num_latent_vars]
            num_samples: number of samples to generate

        Returns:
            x_sample: sampled observed data, shape [B, output_dim, num_samples]
        """

        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            )  # shape [1, num_latent_vars]

        logits_x = self.net(h)  # [B, output_dim]
        probs_x = torch.sigmoid(logits_x)  # [B, output_dim]

        # Generate samples
        x_samples = torch.bernoulli(
            probs_x.unsqueeze(-1).expand(
                -1, -1, num_samples
            )  # [B, output_dim, num_samples]
        )
        return x_samples # [B, output_dim, num_samples]

    def forward(self, x, h):
        """Compute negative log joint probability as loss

        Args:
            x: observed data
            h: latent variables

        Returns:
            loss: negative log joint probability
        """
        log_joint = self.log_joint_prob(x, h)  # [B, num_samples]
        return -log_joint.mean()


class JointModelBernoulliGaussian(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Bernoulli prior for p(h) and Gaussian likelihood for p(x|h).
    """

    SIGMA = 0.1  # Fixed standard deviation for Gaussian likelihood

    def __init__(
        self,
        num_latent_vars=256,
        layers=[512, 512],
        output_dim=784,
        activation: str = "relu",
        final_activation: str = None,
    ):
        super().__init__()

        self._latent_dim = num_latent_vars
        self.num_latent_vars = num_latent_vars
        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=num_latent_vars,
            layers=layers,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation,
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def post_process(self, x):
        """Post-process output x to be in [0, 1] range"""
        if self.net[-1].__class__ == nn.Sigmoid:
            return x  # already in [0, 1]
        elif self.net[-1].__class__ == nn.Tanh:
            return (x + 1) / 2  # scale from [-1, 1] to [0, 1]
        else:
            return torch.clamp(x, 0.0, 1.0)  # clamp to [0, 1]

    def log_prior_prob(self, h):
        """
        Compute log p(h)
        Assuming uniform prior over Bernoulli latent variables
        """
        return h.size(1) * torch.log(torch.tensor(0.5, device=h.device))

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)

        Args:
            x: observed data shape [B, output_dim]
            h: latent variables shape [B, num_latent_vars, num_samples]
            
        Returns:
            log p(x, h): shape [B, num_samples]
            

        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Gaussian distribution for each pixel with fixed variance (e.g., 1.0)
        B, num_latent_vars, num_samples = h.size()
        h_reshaped = h.transpose(1, 2).reshape(
            -1, num_latent_vars
        )  # [B * num_samples, num_latent_vars]
        
        mean_x = self.net(h_reshaped)  # [B * num_samples, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]
        mean_x = mean_x.view(B, num_samples, -1)  # [B, num_samples, output_dim]
        x = x.unsqueeze(1)  # [B, 1, output_dim]
        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        log_p_x_given_h = gaussian_dist.log_prob(x).sum(
            dim=2
        )  # sum over dimensions, shape [B, num_samples]
        return log_p_h + log_p_x_given_h  # log p(x, h), shape [B, num_samples]

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)

        Args:
            h: latent variables shape [B, num_latent_vars]
            num_samples: number of samples to generate

        Returns:
            x_sample: sampled observed data, shape [B, output_dim, num_samples]
        """

        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            ) # shape [1, num_latent_vars]
        mean_x = self.net(h)  # [B, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        x_samples = gaussian_dist.sample((num_samples,)) # [num_samples, B, output_dim]
        x_samples = torch.clamp(x_samples, 0.0, 1.0).permute(1, 2, 0)  # [B, output_dim, num_samples]

        return x_samples # [B, output_dim, num_samples]

    def forward(self, x, h):
        """Compute negative log joint probability as loss

        Args:
            x: observed data, shape [B, output_dim]
            h: latent variables, shape [B, num_latent_vars, num_samples]

        Returns:
            loss: negative log joint probability
        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Gaussian distribution for each pixel with fixed variance (e.g., 1.0)
        B, num_latent_vars, num_samples = h.size()
        h_reshaped = h.transpose(1, 2).reshape(
            -1, num_latent_vars
        )  # [B * num_samples, num_latent_vars]
        
        mean_x = self.net(h_reshaped)  # [B * num_samples, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]
        mean_x = mean_x.view(B, num_samples, -1)  # [B, num_samples, output_dim]
        x = x.unsqueeze(1)  # [B, 1, output_dim]

        # Using MSE loss as negative log likelihood
        mse_loss = nn.MSELoss(reduction="none")
        log_p_x_given_h = -mse_loss(mean_x, x).sum(dim=2)  # shape [B, num_samples]
        log_joint = log_p_h + log_p_x_given_h
        return -log_joint.mean() # negative log joint probability


class JointModelCategoricalGaussian(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Categorical prior for p(h) and Gaussian likelihood for p(x|h).
    """

    SIGMA = 0.1  # Fixed standard deviation for Gaussian likelihood

    def __init__(
        self,
        num_categories,
        num_latent_vars,
        embedding_dims=None,
        layers=[512, 512],
        output_dim=784,
        activation: str = "relu",
        final_activation: str = None,
    ):
        super().__init__()

        self.num_latent_vars = num_latent_vars
        self.output_dim = output_dim

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
        self.net = build_mlp(
            input_dim=self._latent_dim,
            layers=layers,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation,
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def post_process(self, x):
        """Post-process output x to be in [0, 1] range"""
        if self.net[-1].__class__ == nn.Sigmoid:
            return x  # already in [0, 1]
        elif self.net[-1].__class__ == nn.Tanh:
            return (x + 1) / 2  # scale from [-1, 1] to [0, 1]
        else:
            return torch.clamp(x, 0.0, 1.0)  # clamp to [0, 1]

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

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)

        Args:
            x: observed data, shape [B, output_dim]
            h: latent variables, shape [B, num_latent_vars, num_samples]

        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Gaussian distribution for each pixel with fixed variance (e.g., 1.0)
        # Shape of h: [B, num_latent_vars, num_samples]
        h_embedded = [
            embedding(h[:, i, :].long()) # [B, num_samples] -> [B, num_samples, embedding_dim]
            for i, embedding in enumerate(self.embeddings)
        ]
        h_densed = torch.cat(h_embedded, dim=-1)  # [B, num_samples, latent_dim]
        B, num_samples, _ = h_densed.size()
        
        h_reshaped = h_densed.reshape(
            -1, self.latent_dim
        )  # [B * num_samples, latent_dim]

        mean_x = self.net(h_reshaped)  # [B * num_samples, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]
        mean_x = mean_x.view(B, num_samples, self.output_dim)  # [B, num_samples, output_dim]

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        log_p_x_given_h = gaussian_dist.log_prob(x.unsqueeze(1)).sum(
            dim=2
        )  # sum over dimensions, shape [B, num_samples]

        return log_p_h + log_p_x_given_h  # log p(x, h)

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)

        Args:
            h: latent variables
            num_samples: number of samples to generate

        Returns:
            x_sample: sampled observed data
        """

        if h is None:
            # Sample h from uniform categorical distribution
            h_indices = [
                torch.randint(0, K, (1,)).to(next(self.parameters()).device)
                for K in self._num_categories
            ]
            h = torch.cat(h_indices, dim=-1)  # shape [1, num_latent_vars]

        h_coded = [
            embedding(h[:, i].long())  # [B, num_latent_vars] -> [B, embedding_dim]
            for i, embedding in enumerate(self.embeddings)
        ]
        h_densed = torch.cat(h_coded, dim=-1)  # [B, latent_dim]
        
        mean_x = self.net(h_densed)  # [B, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        x_samples = gaussian_dist.sample((num_samples,))
        x_samples = torch.clamp(x_samples, 0.0, 1.0).permute(1, 2, 0)  # [B, output_dim, num_samples]
        return x_samples  # [B, output_dim, num_samples]

    def forward(self, x, h):
        """Compute negative log joint probability as loss

        Args:
            x: observed data
            h: latent variables

        Returns:
            loss: negative log joint probability
        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Gaussian distribution for each pixel with fixed variance (e.g., 1.0)
        B, num_latent_vars, num_samples = h.size()
        h_embedded = [
            embedding(h[:, i, :].long()) # [B, num_samples] -> [B, num_samples, embedding_dim]
            for i, embedding in enumerate(self.embeddings)
        ]
        h_densed = torch.cat(h_embedded, dim=-1)  # [B, num_samples, latent_dim]
        h_reshaped = h_densed.reshape(
            -1, self.latent_dim
        )  # [B * num_samples, latent_dim]
        
        mean_x = self.net(h_reshaped)  # [B * num_samples, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]
        mean_x = mean_x.view(B, num_samples, self.output_dim)  # [B, num_samples, output_dim]
        x = x.unsqueeze(1)  # [B, 1, output_dim]
        # Using MSE loss as negative log likelihood
        mse_loss = nn.MSELoss(reduction="none")
        log_p_x_given_h = -mse_loss(mean_x, x).sum(dim=2)  # shape [B, num_samples]
        log_p_x_given_h = log_p_x_given_h.mean(dim=1)  # average over samples, shape [B]
        log_joint = log_p_h + log_p_x_given_h
        return -log_joint.mean()
