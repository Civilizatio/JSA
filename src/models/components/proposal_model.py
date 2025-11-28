# src/models/components/proposal_model.py
import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseProposalModel
from src.utils.mlp_utils import build_mlp

class ProposalModel(BaseProposalModel):
    """
    q_phi(h|x), must implement:
        - log_conditional_prob(h, x)
        - sample_latent(x)
    """
    def __init__(self, input_dim=784, layers=[512, 512], latent_dim=256, activation: str = "relu"):
        super().__init__()
        
        self._latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.net = build_mlp(
            input_dim=input_dim, layers=layers, output_dim=latent_dim, activation=activation
        )
        
    @property
    def latent_dim(self):
        return self._latent_dim
        
    def sample_latent(self, x, num_samples=1):
        """ Sample h ~ q(h|x)"""
        logits = self.net(x) # [B, latent_dim]
        probs = torch.sigmoid(logits)
        h_samples = torch.bernoulli(probs.repeat(num_samples, 1))  # [num_samples*B, latent_dim]
        return h_samples

    def log_conditional_prob(self, h, x):
        logits = self.net(x) # [B, latent_dim]
        return -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, h, reduction="none"
        ).sum(dim=1)
        
    def forward(self, h, x):
        """Compute negative log conditional probability as loss

        Args:
            h: latent variables
            x: observed data

        Returns:
            loss: negative log conditional probability
        """
        log_cond = self.log_conditional_prob(h, x)
        return -log_cond.mean()
