# src/samplers/ncg_sampler.py
"""A practical Norm-Constrained Gradient (NCG) sampler for discrete latents.

This implementation follows the spirit of gradient-guided local proposals on token embeddings:

1. compute the distortion gradient with respect to the current latent embeddings;
2. build a proposal distribution over nearby token embeddings at a few selected sites;
3. accept or reject the proposal with a Metropolis-Hastings correction using the *full* joint energy.

The proposal uses only the distortion gradient by default. This keeps the implementation broadly
compatible with the current decoder-centric setup while still allowing the acceptance ratio to account
for the prior energy.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from src.base.base_sampler import BaseSampler



class NCGSampler(BaseSampler):
    """ Norm Constrained Gradient (NCG) sampler for discrete latents. 
    
    This sampler uses the distortion gradient to propose local changes to the latent tokens, 
    and then applies a Metropolis-Hastings acceptance step based on the full joint energy. 
    The proposal distribution is designed to favor tokens that reduce distortion while also considering a norm constraint to encourage exploration.
    
    Args:
        joint_model: A model that exposes a `log_joint_prob(x, h)` method and a `distortion_model` with a `decoder` that has `embeddings`. This is used to compute the energies and gradients needed for the NCG proposals.
        proposal_model: (Optional) A model used to compute the proposal distribution. Not used in this implementation but included for API consistency and potential future extensions.
        alpha: A scaling factor for the norm constraint in the proposal distribution. Higher values encourage proposals that are closer to the current token embedding.
        p_norm: The order of the norm used in the proposal distribution. Common choices are 1 (L1) or 2 (L2).
        num_steps: The number of NCG steps to perform when `sample` is called.
        num_sites: The number of token positions to update in each proposal. These are selected randomly at each step.
        temperature: A temperature parameter for the proposal distribution. Higher values lead to more uniform proposals, while lower values make the proposal distribution more peaked around the tokens favored by the distortion gradient.
        include_current_token: Whether to include the current token in the proposal distribution. If False, the sampler will only propose changes to different tokens, which can encourage exploration but may also lead to higher rejection rates.
    
    The proposal should be:
    q(h' | h) \propto \exp\left( -\frac{1}{2} \nabla_{h} D(x, h) \cdot (e(h') - e(h)) - \frac{1}{2\alpha} \|e(h') - e(h)\|_p^p \right)
    
    """
    
    def __init__(
        self,
        joint_model: nn.Module,
        proposal_model: Optional[nn.Module] = None, # Not used in this implementation but included for API consistency/future extensions
        alpha: float = 1.0,
        p_norm: float = 2.0,
        num_steps: int = 1,
        num_sites: int = 1,
        temperature: float = 1.0,
        include_current_token: bool = True,
    ):
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.alpha = float(alpha)
        self.p_norm = float(p_norm)
        self.num_steps = int(num_steps)
        self.num_sites = int(num_sites)
        self.temperature = float(temperature)
        self.include_current_token = bool(include_current_token)

    def to(self, device: torch.device):
        return self

    def _prepare_h(self, h: Tensor) -> Tensor:
        """Prepares the latent tensor `h` for NCG proposals. If `h` has a single-channel spatial structure, it is treated as token indices and returned as-is. 
        Otherwise, it is treated as embedded latents and returned as-is for gradient computation."""
        if h.dim() == 5 and h.shape[1] == 1: # [B, 1, H, W, num_latent_vars]
            return h[:, 0].clone() # [B, H, W, num_latent_vars] treated as token indices
        if h.dim() < 3:
            raise ValueError(
                f"NCGSampler expects h with spatial dimensions, got shape {tuple(h.shape)}."
            )
        return h.clone()

    def _get_decoder_and_embeddings(self):
        distortion_model = getattr(self.joint_model, "distortion_model", None)
        if distortion_model is None:
            raise AttributeError("joint_model must expose `distortion_model` for NCGSampler.")
        decoder = getattr(distortion_model, "decoder", None)
        if decoder is None or not hasattr(decoder, "embeddings"):
            raise AttributeError(
                "The distortion decoder must expose `embeddings` so that NCG proposals can be computed."
            )
        return distortion_model, decoder

    def _embedding_gradients(self, x: Tensor, h: Tensor) -> Tensor:
        """ Computes the gradient of the distortion with respect to the latent embeddings. 
        This is used to guide the NCG proposals."""
        distortion_model, _ = self._get_decoder_and_embeddings()
        embedded_h = distortion_model.embed_latent(h).detach().requires_grad_(True)
        x_hat = distortion_model.decode_from_embedded_latent(embedded_h)
        out = distortion_model.distortion_from_reconstruction(x, x_hat)
        grad = torch.autograd.grad(out.distortion.sum(), embedded_h, only_inputs=True)[0]
        return grad

    def _site_scores(
        self,
        embedding_table: Tensor,
        grad_site: Tensor,
        current_idx: Tensor,
        current_embed: Tensor,
    ) -> Tensor:
        """Computes unnormalized log-probability scores for all tokens in the embedding table at a single site, 
        given the distortion gradient at that site and the current token embedding."""
        
        # embedding_table: [K, D]
        # grad_site/current_embed/current_idx batch-aligned.
        diff = embedding_table.unsqueeze(0) - current_embed.unsqueeze(1)
        linear_term = -0.5 * torch.sum(grad_site.unsqueeze(1) * diff, dim=-1)
        norm_term = -0.5 / self.alpha * torch.sum(diff.abs().pow(self.p_norm), dim=-1)
        scores = linear_term + norm_term
        if self.include_current_token:
            scores.scatter_(1, current_idx.unsqueeze(1), scores.gather(1, current_idx.unsqueeze(1)))
        scores = scores / max(self.temperature, 1e-6)
        return scores

    def _flatten_index(self, h: Tensor, latent_var: int):
        current_tokens = h[..., latent_var].long()
        return current_tokens.reshape(h.shape[0], -1)

    def _split_gradients(self, grad: Tensor, decoder) -> List[Tensor]:
        dims = list(getattr(decoder, "embedding_dims", []))
        if not dims:
            raise AttributeError(
                "Decoder must define `embedding_dims` for multi-variable NCG proposals."
            )
        return list(torch.split(grad, dims, dim=-1))

    def _propose(self, x: Tensor, h: Tensor):
        distortion_model, decoder = self._get_decoder_and_embeddings()
        grad = self._embedding_gradients(x, h)
        grad_splits = self._split_gradients(grad, decoder)

        proposal_log_prob = torch.zeros(h.shape[0], device=h.device)
        h_new = h.clone()
        selected = []

        num_positions = h[..., 0].numel() // h.shape[0]
        num_sites = min(self.num_sites, num_positions)
        position_ids = torch.randperm(num_positions, device=h.device)[:num_sites]

        for latent_var, (embedding, grad_chunk) in enumerate(zip(decoder.embeddings, grad_splits)):
            token_grid = self._flatten_index(h_new, latent_var)
            grad_grid = grad_chunk.reshape(h.shape[0], num_positions, -1)
            embedding_weight = embedding.weight.detach()

            for pos in position_ids.tolist():
                cur_idx = token_grid[:, pos]
                cur_embed = embedding_weight[cur_idx]
                grad_site = grad_grid[:, pos]
                scores = self._site_scores(embedding_weight, grad_site, cur_idx, cur_embed)
                probs = F.softmax(scores, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                proposal_log_prob = proposal_log_prob + torch.log(
                    probs.gather(1, sampled_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
                )
                token_grid[:, pos] = sampled_idx
                selected.append((latent_var, pos))

            h_new[..., latent_var] = token_grid.view_as(h_new[..., latent_var]).float()

        return h_new, proposal_log_prob, selected

    def _reverse_log_prob(self, x: Tensor, h_from: Tensor, h_to: Tensor, selected):
        distortion_model, decoder = self._get_decoder_and_embeddings()
        grad = self._embedding_gradients(x, h_to)
        grad_splits = self._split_gradients(grad, decoder)

        num_positions = h_to[..., 0].numel() // h_to.shape[0]
        reverse_log_prob = torch.zeros(h_to.shape[0], device=h_to.device)

        for latent_var, pos in selected:
            embedding = decoder.embeddings[latent_var]
            embedding_weight = embedding.weight.detach()
            grad_grid = grad_splits[latent_var].reshape(h_to.shape[0], num_positions, -1)
            token_grid_to = h_to[..., latent_var].long().reshape(h_to.shape[0], num_positions)
            token_grid_from = h_from[..., latent_var].long().reshape(h_from.shape[0], num_positions)
            cur_idx = token_grid_to[:, pos]
            old_idx = token_grid_from[:, pos]
            cur_embed = embedding_weight[cur_idx]
            grad_site = grad_grid[:, pos]
            scores = self._site_scores(embedding_weight, grad_site, cur_idx, cur_embed)
            probs = F.softmax(scores, dim=-1)
            reverse_log_prob = reverse_log_prob + torch.log(
                probs.gather(1, old_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            )

        return reverse_log_prob

    def step(self, x: Tensor, h: Tensor) -> Tensor:
        h_old = self._prepare_h(h)
        h_new, log_q_forward, selected = self._propose(x, h_old)
        log_q_reverse = self._reverse_log_prob(x, h_old, h_new, selected)

        log_p_old = self.joint_model.log_joint_prob(x, h_old)
        log_p_new = self.joint_model.log_joint_prob(x, h_new)
        log_accept = (log_p_new + log_q_reverse) - (log_p_old + log_q_forward)
        accept_prob = torch.exp(torch.clamp(log_accept, max=0.0, min=-100.0))
        accept = torch.rand_like(accept_prob) < accept_prob
        accept_view = accept.view(-1, *([1] * (h_old.dim() - 1)))
        h_next = torch.where(accept_view, h_new, h_old)
        return h_next.float()

    def sample(self, x: Tensor, h_init: Tensor, num_steps: Optional[int] = None, return_all: bool = False):
        steps = self.num_steps if num_steps is None else int(num_steps)
        h = self._prepare_h(h_init)
        traj = []
        for _ in range(steps):
            h = self.step(x, h)
            if return_all:
                traj.append(h)
        if return_all:
            return torch.stack(traj, dim=1)
        return h


__all__ = ["NCGSampler"]
