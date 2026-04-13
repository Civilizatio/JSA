# src/models/components/joint_model.py
from __future__ import annotations
import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseJointModel
import math


# // This class has been deprecated, as we found that Bernoulli likelihood is too hard to optimize for complex datasets like ImageNet, even with a powerful network. We keep it here for reference and potential future use on simpler datasets.
import warnings


class _JointModelBernoulliBernoulli(BaseJointModel):
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
        warnings.warn(
            "JointModelBernoulliBernoulli is deprecated and may not perform well on complex datasets. Consider using JointModelCategoricalGaussian instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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


# // Remove Bernoulli likelihood model and use Categorical-Gaussian model instead, as Bernoulli likelihood is too hard to optimize for complex datasets like ImageNet, even with a powerful network. We keep it here for reference and potential future use on simpler datasets.
class _JointModelBernoulliGaussian(BaseJointModel):
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
        warnings.warn(
            "The _JointModelBernoulliGaussian model is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        return torch.clamp(x_samples, -1.0, 1.0)

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
        sigma=0.1,
        sigma_mode: str = "learnable",  # whether sigma is learnable or fixed or scheduled
        sample_chunk_size=8,
    ):
        super().__init__()

        self.sample_chunk_size = (
            sample_chunk_size  # for sampling in chunks to save memory
        )
        self.sigma_mode = sigma_mode
        self.net = net

        init_log_sigma = torch.log(torch.tensor(float(sigma)))
        if self.sigma_mode == "learnable":
            self.log_sigma = nn.Parameter(init_log_sigma.clone())
        elif self.sigma_mode in ["fixed", "scheduled"]:
            self.register_buffer("log_sigma", init_log_sigma.clone())
        else:
            raise ValueError(f"Invalid sigma_mode: {self.sigma_mode}")

    @property
    def latent_dim(self):
        return self.net.latent_dim

    @property
    def num_categories(self):
        return self.net.num_categories

    @property
    def num_latent_vars(self):
        return self.net.num_latent_vars

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def set_sigma(self, sigma):
        """Only used when sigma is fixed or scheduled, not learnable"""
        if self.sigma_mode == "learnable":
            raise ValueError("Cannot set sigma when it is learnable")

        with torch.no_grad():
            self.log_sigma.fill_(math.log(sigma))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def log_prior_prob(self, h):
        """
        Compute log p(h)
        Assuming uniform prior over Categorical latent variables
        Mathmatically, for each latent variable with K categories, log p(h_i) = -log(K)
        Thus, log p(h) = sum over latent variables of -log(K_i)
        """
        num_categories_tensor = torch.tensor(
            self.num_categories, device=h.device, dtype=torch.float
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

        # Methods below may cause OOM for large num_samples, so we do chunking
        # Flatten first two dimensions to compute log_joint_prob
        # h_flat = h.reshape(
        #     batch_size * num_samples, *h.shape[2:]
        # )  # [B * num_samples, ..., num_latent_vars]
        # log_probs_flat = self.log_joint_prob(
        #     x.repeat_interleave(num_samples, dim=0), h_flat
        # )  # [B * num_samples]
        # log_probs = log_probs_flat.view(batch_size, num_samples)  # [B, num_samples]

        log_probs_list = []
        for i in range(0, num_samples, self.sample_chunk_size):
            chunk_size = min(self.sample_chunk_size, num_samples - i)
            h_chunk = h[
                :, i : i + chunk_size, ...
            ]  # [B, chunk_size, ..., num_latent_vars]
            h_chunk_flat = h_chunk.reshape(
                batch_size * chunk_size, *h_chunk.shape[2:]
            )  # [B * chunk_size, ..., num_latent_vars]
            x_expanded = x.repeat_interleave(chunk_size, dim=0)  # [B * chunk_size, ...]
            log_probs_chunk_flat = self.log_joint_prob(
                x_expanded, h_chunk_flat
            )  # [B * chunk_size]
            log_probs_chunk = log_probs_chunk_flat.view(
                batch_size, chunk_size
            )  # [B, chunk_size]
            log_probs_list.append(log_probs_chunk)

        log_probs = torch.cat(log_probs_list, dim=1)  # [B, num_samples]
        return log_probs  # [B, num_samples]

    def forward_multiple_samples(self, h):
        """Compute p(x|h) parameters (mean) for multiple samples of h

        Args:
            h: latent variables, shape [B, num_samples, ..., num_latent_vars]
        Returns:
            mean_x: shape [B, num_samples, ...]
        """
        batch_size, num_samples = h.shape[0], h.shape[1]

        # Methods below may cause OOM for large num_samples, so we do chunking
        # # Flatten first two dimensions to compute forward
        h_flat = h.reshape(
            batch_size * num_samples, *h.shape[2:]
        )  # [B * num_samples, ..., num_latent_vars]
        mean_x_flat = self.forward(h_flat)  # [B * num_samples, ...]
        mean_x = mean_x_flat.view(
            batch_size, num_samples, *mean_x_flat.shape[1:]
        )  # [B, num_samples, ...]

        # Chunking to save memory
        # outputs = []
        # for i in range(0, num_samples, self.sample_chunk_size):
        #     h_chunk = h[:, i : i + self.sample_chunk_size, ...]  # [B, chunk_size, ..., num_latent_vars]
        #     chunk_size = h_chunk.shape[1]
        #     h_chunk_flat = h_chunk.reshape(
        #         batch_size * chunk_size, *h_chunk.shape[2:]
        #     )  # [B * chunk_size, ..., num_latent_vars]
        #     mean_x_chunk_flat = self.forward(h_chunk_flat)  # [B * chunk_size, ...]
        #     mean_x_chunk = mean_x_chunk_flat.view(
        #         batch_size, chunk_size, *mean_x_chunk_flat.shape[1:]
        #     )  # [B, chunk_size, ...]
        #     outputs.append(mean_x_chunk)

        # mean_x = torch.cat(outputs, dim=1)  # [B, num_samples, ...]
        return mean_x  # [B, num_samples, ...]

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

        mu_new = mu_new.view(
            batch_size, num_samples, *mu_new.shape[1:]
        )  # [B, num_samples, ...]
        mu_old = mu_old.view(
            batch_size, num_samples, *mu_old.shape[1:]
        )  # [B, num_samples, ...]

        x_expanded = x.unsqueeze(1)

        # Using the theoretical form:
        # -0.5/sigma^2 * ( ||x - mu_new||^2 - ||x - mu_old||^2 )
        # 1. Squared errors
        se_new = (x_expanded - mu_new) ** 2  # [B, num_samples, ...]
        se_old = (x_expanded - mu_old) ** 2  # [B, num_samples, ...]
        # 2. Sum over data dimensions
        sse_new = se_new.sum(dim=list(range(2, se_new.dim())))  # [B, num_samples]
        sse_old = se_old.sum(dim=list(range(2, se_old.dim())))  # [B, num_samples]
        # 3. Difference
        log_prob_diff = (
            -0.5 / (self.sigma**2) * (sse_new - sse_old)
        )  # shape [B, num_samples]

        # Below is an equivalent form that is more likely to cause numerical issues
        # diff =  (mu_new-mu_old)*(x_expanded - 0.5*(mu_new + mu_old))/(self.sigma**2)
        # log_prob_diff = diff.sum(dim=list(range(2, diff.dim())))  # sum over dimensions, shape [B, num_samples]

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
            torch.clamp(x_samples, -1.0, 1.0)
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

    def get_loss(self, x, h, return_forward=False, backward_fn=None) -> JointLossOutput:
        """Compute negative log joint probability as loss

        Args:
            x: observed data, shape [B, ...]
            h: latent variables, shape [B, num_samples, ..., num_latent_vars]
            return_forward: whether to return the forward output (mean_x),
                            if multiple samples are used, only return the last sample's output
            backward_fn: optional backward function to apply gradients

        Returns:
            loss: negative log joint probability
            mean_x: decoded observed data, shape [N, ...] (only if return_forward is True)
        """

        # mean_x = self.forward_multiple_samples(h)  # [B, num_samples, ...]
        # x = x.unsqueeze(1)  # [B, 1, ...], broadcast to [B, num_samples, ...]
        batch_size, num_samples = h.shape[0], h.shape[1]
        mse = torch.nn.MSELoss(reduction="mean")
        D = x.numel() / batch_size  # dimensionality of x
        total_loss = 0.0
        if return_forward:
            mean_x_last = None

        for i in range(0, num_samples, self.sample_chunk_size):
            chunk_size = min(self.sample_chunk_size, num_samples - i)
            h_chunk = h[
                :, i : i + chunk_size, ...
            ]  # [B, chunk_size, ..., num_latent_vars]
            mean_x_chunk = self.forward_multiple_samples(
                h_chunk
            )  # [B, chunk_size, ...]

            x_expanded = x.unsqueeze(1).expand_as(mean_x_chunk)  # [B, chunk_size, ...]
            # Loss should be:
            # 1/(2*sigma^2) * ||x - mean_x||^2 + D/2 * log(2*pi*sigma^2)
            # where D is the dimensionality of x
            # Here, sigma is a learnable parameter, we cannot ignore the latter term
            # So the loss becomes:
            # mse = nn.MSELoss(reduction='mean')
            loss_mse = mse(
                mean_x_chunk, x_expanded
            )  # mean squared error over all dimensions
            # Compute loss for the chunk
            loss_chunk = loss_mse / (2 * self.sigma**2) + torch.log(
                self.sigma
            )  # mean over all dimensions
            # loss_chunk = mse(mean_x_chunk, x_expanded)/(2 * sigma**2)  # mean over all dimensions

            # backward if function provided
            if backward_fn is not None:
                backward_fn(loss_chunk)

            total_loss += (
                loss_mse.detach() * chunk_size
            )  # sum up the loss chunks weighted by chunk size
        else:
            if return_forward and mean_x_last is None:
                mean_x_last = mean_x_chunk[
                    :, -1:, ...
                ]  # last sample in the last chunk, shape [B, 1, ...]

        total_loss = total_loss / num_samples  # average over all samples

        return JointLossOutput(
            loss=total_loss,
            reconstruction=mean_x_last.squeeze(1) if return_forward else None
        )

    def forward(self, h):
        """Decode x ~ p(x|h)

        Args:
            h: latent variables, shape [N, ..., num_latent_vars]

        Returns:
            mean_x: decoded observed data, shape [N, ...]
        """
        mean_x = self.net(h)  # [N, ...]
        return mean_x  # [N, ...]


from src.modules.losses.perceptual_distortion import (
    ReconstructionDistortionModel,
    ReconstructionDistortionForwardOutput,
)
from src.modules.jsa.prior_model import UniformPriorEnergy
from src.utils.perceptual_stage import PerceptualTrainingStage
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class JointLossOutput:
    loss: torch.Tensor
    reconstruction: Optional[torch.Tensor] = None
    components: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class EnergyOutput:
    energy: torch.Tensor
    reconstruction: torch.Tensor
    components: Dict[str, torch.Tensor]


class JointModelCategoricalEnergy(BaseJointModel):
    """Energy-based joint model for perceptual JSA.

    The model keeps the original JSA view of a *joint* score over ``(x, h)`` while replacing the
    Gaussian likelihood with a more general energy decomposition:

    ``u(x, h) = lambda_d * d(x, h) + lambda_p * f(h)``

    where ``d`` is the reconstruction/perceptual distortion and ``f`` is a prior energy.

    Stage behaviour
    ---------------
    * stage 1 / stage 2: only the distortion energy participates in ``log p(x, h)``;
    * stage 3: both distortion and prior energies are active.

    This makes stage-2 latent sampling consistent with the task description: the latent codes used
    to pretrain the prior are still sampled from the stage-1 distortion-only posterior.
    """

    def __init__(
        self,
        distortion_model: ReconstructionDistortionModel,
        prior_model: nn.Module = None,
        distortion_weight: float = 1.0,
        prior_weight: float = 1.0,
        distortion_temperature: float = 1.0,
        temperature_mode: str = "fixed",
        sample_chunk_size: int = 8,
        default_stage=PerceptualTrainingStage.FULL_EBM,
    ):
        super().__init__()
        self.distortion_model = distortion_model
        self.prior_model = (
            prior_model
            if prior_model is not None
            else UniformPriorEnergy(
                num_categories=getattr(distortion_model, "num_categories", None)
            )
        )
        self.distortion_weight = float(distortion_weight)
        self.prior_weight = float(prior_weight)
        self.sample_chunk_size = int(sample_chunk_size)
        self.current_stage = PerceptualTrainingStage.from_value(default_stage)

        if temperature_mode == "scheduled":
            raise NotImplementedError("scheduled temperature mode is not supported yet.")
        self.temperature_mode = temperature_mode
        init_log_T = torch.log(torch.tensor(float(distortion_temperature)))
        if self.temperature_mode == "learnable":
            self.log_distortion_temperature = nn.Parameter(init_log_T)
        elif self.temperature_mode == "fixed":
            self.register_buffer("log_distortion_temperature", init_log_T)
        else:
            raise ValueError(f"Unknown temperature_mode: {temperature_mode}")

        # Keep a `net` attribute for backward compatibility with callbacks and summary code.
        self.net = getattr(self.distortion_model, "decoder", self.distortion_model)

    @property
    def distortion_temperature(self):
        return torch.exp(self.log_distortion_temperature)

    @property
    def latent_dim(self):
        return getattr(self.distortion_model, "latent_dim", None)

    @property
    def num_categories(self):
        return getattr(self.distortion_model, "num_categories", None)

    def set_stage(self, stage):
        self.current_stage = PerceptualTrainingStage.from_value(stage)

    def _resolve_stage(self, stage=None):
        stage = PerceptualTrainingStage.from_value(stage)
        if stage is None:
            stage = self.current_stage
        return stage

    def _component_weights(self, stage=None):
        stage = self._resolve_stage(stage)
        distortion_weight = self.distortion_weight
        if stage in {
            PerceptualTrainingStage.DECODER_PRETRAIN,
            PerceptualTrainingStage.PRIOR_PRETRAIN,
        }:
            prior_weight = 0.0
        else:
            prior_weight = self.prior_weight
        return distortion_weight, prior_weight

    def get_last_layer_weight(self):
        if hasattr(self.distortion_model, "get_last_layer_weight"):
            return self.distortion_model.get_last_layer_weight()
        return None

    def forward(self, h):
        return self.distortion_model.decode(h)

    def decode(self, h):
        return self.forward(h)

    def distortion(self, x, h) -> ReconstructionDistortionForwardOutput:
        x_hat = self.forward(h)
        out = self.distortion_model.distortion_from_reconstruction(x, x_hat)
        return ReconstructionDistortionForwardOutput(
            distortion=out.distortion, reconstruction=x_hat, components=out.components
        )

    def prior_energy(self, h):
        return self.prior_model(h)

    def energy(self, x, h, stage=None) -> EnergyOutput:
        dist_out = self.distortion(x, h)
        prior = self.prior_energy(h)
        distortion_weight, prior_weight = self._component_weights(stage)
        
        total = distortion_weight * (dist_out.distortion / self.distortion_temperature) + prior_weight * prior

        components = {
            **dist_out.components,
            "prior_total": prior,
            "energy_total": total,
            "distortion_weight": torch.full_like(total, distortion_weight),
            "prior_weight": torch.full_like(total, prior_weight),
            "distortion_temperature": self.distortion_temperature.expand_as(total),
        }
        return EnergyOutput(
            energy=total, reconstruction=dist_out.reconstruction, components=components
        )

    def log_joint_prob(self, x, h, stage=None):
        return -self.energy(x, h, stage=stage).energy

    def log_joint_prob_multiple_samples(self, x, h, stage=None):
        batch_size, num_samples = h.shape[0], h.shape[1]
        outputs = []
        for i in range(0, num_samples, self.sample_chunk_size):
            chunk_size = min(self.sample_chunk_size, num_samples - i)
            h_chunk = h[:, i : i + chunk_size, ...]
            h_flat = h_chunk.reshape(batch_size * chunk_size, *h_chunk.shape[2:])
            x_expanded = x.repeat_interleave(chunk_size, dim=0)
            chunk = self.log_joint_prob(x_expanded, h_flat, stage=stage)
            outputs.append(chunk.view(batch_size, chunk_size))
        return torch.cat(outputs, dim=1)

    def log_joint_prob_diff(self, x, h_new, h_old, stage=None):
        return self.log_joint_prob_multiple_samples(
            x, h_new, stage=stage
        ) - self.log_joint_prob_multiple_samples(x, h_old, stage=stage)

    def energy_multiple_samples(self, x, h, stage=None) -> EnergyOutput:
        batch_size, num_samples = h.shape[0], h.shape[1]
        energy_list = []
        recon_list = []
        component_buckets = {}

        for i in range(0, num_samples, self.sample_chunk_size):
            chunk_size = min(self.sample_chunk_size, num_samples - i)
            h_chunk = h[:, i : i + chunk_size, ...]
            h_flat = h_chunk.reshape(batch_size * chunk_size, *h_chunk.shape[2:])
            x_expanded = x.repeat_interleave(chunk_size, dim=0)

            out = self.energy(x_expanded, h_flat, stage=stage)

            energy_list.append(out.energy.view(batch_size, chunk_size))
            recon_list.append(
                out.reconstruction.view(
                    batch_size, chunk_size, *out.reconstruction.shape[1:]
                )
            )

            if not component_buckets:
                component_buckets = {key: [] for key in out.components}
            for key, value in out.components.items():
                component_buckets[key].append(value.view(batch_size, chunk_size))

        energy_all = torch.cat(energy_list, dim=1)
        recon_all = torch.cat(recon_list, dim=1)
        components_all = {
            key: torch.cat(values, dim=1) for key, values in component_buckets.items()
        }

        return EnergyOutput(
            energy=energy_all, reconstruction=recon_all, components=components_all
        )

    def sample(self, h=None, num_samples=1, stage=None):
        """Sample x ~ p(x|h) by finding the lowest energy reconstruction for the given h.
        
        NOTE: This sampling method is deterministic and only returns the mean reconstruction, which is not a true sample from the distribution.
        """
        if h is None:
            if hasattr(self.prior_model, "sample") and self.num_categories is not None:
                raise ValueError(
                    "Please provide `h` explicitly when using the energy model sample() helper. "
                    "Sampling from the prior should be performed via `prior_model.sample(...)`."
                )
            raise ValueError(
                "`h` must be provided for JointModelCategoricalEnergy.sample`."
            )
        x_hat = self.decode(h)
        if num_samples <= 1:
            return x_hat.unsqueeze(1)
        return x_hat.unsqueeze(1).expand(-1, num_samples, *x_hat.shape[1:]).contiguous()

    def get_loss(
        self,
        x,
        h,
        stage=None,
        return_forward: bool = False,
        return_components: bool = False,
        backward_fn=None,
        normalize_loss_by_dim: bool = True,  # Added: Divide loss by data dimensionality
    ) -> JointLossOutput:
        # We fetch full output using dataclass and then unpack as caller expects
        outputs = self.energy_multiple_samples(x, h, stage=stage)

        energy_all = outputs.energy
        recon_all = outputs.reconstruction
        component_all = outputs.components

        loss = energy_all.mean()
        resolved_stage = self._resolve_stage(stage)
        
        # When pixel parameters use reduction='sum' for strict probabilistic correctness 
        # during sampling, the resulting loss scale is immense. 
        # Dividing by data dimensionality decouples the optimization scale (learning rate)
        # from the mathematically rigid energy landscape.
        D = x.numel() / x.shape[0] if x.shape[0] > 0 else 1.0

        temperature_regularizer = torch.zeros((), device=x.device, dtype=loss.dtype)
        if (
            self.temperature_mode == "learnable"
            and resolved_stage == PerceptualTrainingStage.DECODER_PRETRAIN
        ):
            # Stage-1 has only positive phase, so this regularizer prevents T from drifting to +inf.
            c = D
            temperature_regularizer = c * self.log_distortion_temperature
            loss = loss + temperature_regularizer

        scale_factor = D if normalize_loss_by_dim else 1.0
        
        loss = loss / scale_factor

        if backward_fn is not None:
            backward_fn(loss)

        recon_out = recon_all[:, -1, ...] if return_forward else None
        
        summary = None
        if return_components:
            summary = {}
            for key, value in component_all.items():
                if key in ["distortion_weight", "prior_weight", "distortion_temperature"]:
                    summary[key] = value.mean()
                else:
                    summary[key] = value.mean() / scale_factor
            summary["temperature_regularizer"] = temperature_regularizer / scale_factor

        return JointLossOutput(
            loss=loss,
            reconstruction=recon_out,
            components=summary
        )
