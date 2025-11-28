# src/models/components/joint_model.py

import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseJointModel
from src.utils.mlp_utils import build_mlp


class JointModelBernoulliBernoulli(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)

    We assume Bernoulli prior for p(h) and Bernoulli likelihood for p(x|h)
    Using MLP to model p_theta(x|h)
    """

    def __init__(
        self,
        latent_dim=256,
        layers=[512, 512],
        output_dim=784,
        activation: str = "relu",
        final_activation: str = None,
    ):
        super().__init__()

        self._latent_dim = latent_dim
        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=latent_dim,
            layers=layers,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation,
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def log_prior_prob(self, h):
        """
        Compute log p(h)
        Assuming uniform prior over Bernoulli latent variables
        """
        # For Bernoulli latent variables with uniform prior,
        # log p(h) = sum over dimensions of log(0.5) = h.size(1) * log(0.5)
        return h.size(1) * torch.log(torch.tensor(0.5, device=h.device))

    def log_joint_prob(self, x, h):
        """Must compute log p(x, h) = log p(h) + log p(x|h)

        Args:
            x: observed data
            h: latent variables

        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Bernoulli distribution for each pixel
        logits_x = self.net(h)
        log_p_x_given_h = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits_x, x, reduction="none"
        ).sum(dim=1)
        return log_p_h + log_p_x_given_h

    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)

        Args:
            h: latent variables
            num_samples: number of samples to generate

        Returns:
            x_sample: sampled observed data
        """

        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            )
        logits_x = self.net(h)
        probs_x = torch.sigmoid(logits_x)

        # Generate samples
        x_samples = torch.bernoulli(
            probs_x.repeat(num_samples, 1)
        )  # [num_samples, output_dim]
        return x_samples
    
    def forward(self, x, h):
        """Compute negative log joint probability as loss

        Args:
            x: observed data
            h: latent variables

        Returns:
            loss: negative log joint probability
        """
        log_joint = self.log_joint_prob(x, h)
        return -log_joint.mean()


class JointModelBernoulliGaussian(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Bernoulli prior for p(h) and Gaussian likelihood for p(x|h).
    """
    SIGMA = 0.1  # Fixed standard deviation for Gaussian likelihood
    
    def __init__(self, latent_dim=256, layers=[512, 512], output_dim=784, activation: str = "relu", final_activation: str = None):
        super().__init__()

        self._latent_dim = latent_dim
        self.output_dim = output_dim
        self.net = build_mlp(
            input_dim=latent_dim, layers=layers, output_dim=output_dim, activation=activation, final_activation=final_activation
        )
        
    @property
    def latent_dim(self):
        return self._latent_dim
    
    def post_process(self, x):
        """ Post-process output x to be in [0, 1] range"""
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
            x: observed data
            h: latent variables

        """
        # Prior p(h)
        log_p_h = self.log_prior_prob(h)

        # Likelihood p(x|h)
        # We assume independent Gaussian distribution for each pixel with fixed variance (e.g., 1.0)
        mean_x = self.net(h) # [B, output_dim]
        mean_x = self.post_process(mean_x) # ensure mean is in [0, 1]
        
        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        log_p_x_given_h = gaussian_dist.log_prob(x).sum(dim=1)  # sum over dimensions, shape [B]

        return log_p_h + log_p_x_given_h # log p(x, h)
    
    def sample(self, h=None, num_samples=1):
        """Sample x ~ p(x|h)

        Args:
            h: latent variables
            num_samples: number of samples to generate

        Returns:
            x_sample: sampled observed data
        """

        if h is None:
            h = torch.bernoulli(0.5 * torch.ones((1, self.latent_dim))).to(
                next(self.parameters()).device
            )
        mean_x = self.net(h)  # [B, output_dim]
        mean_x = self.post_process(mean_x)  # ensure mean is in [0, 1]

        gaussian_dist = torch.distributions.Normal(loc=mean_x, scale=self.SIGMA)
        x_samples = gaussian_dist.sample((num_samples,))  # [num_samples, output_dim]
        x_samples = torch.clamp(x_samples, 0.0, 1.0)  # ensure samples are in [0, 1]
        
        return x_samples
    
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
        mean_x = self.net(h) # [B, output_dim]
        mean_x = self.post_process(mean_x) # ensure mean is in [0, 1]
        
        # Using MSE loss as negative log likelihood
        mse_loss = nn.MSELoss(reduction="none")
        log_p_x_given_h = -mse_loss(mean_x, x).sum(dim=1)  # shape [B]
        log_joint = log_p_h + log_p_x_given_h
        return -log_joint.mean()
