# src/samplers/ncg_sampler.py
"""Norm-Constrained Gradient (NCG) sampler for discrete latent codes.

The sampler follows the score-composition strategy described in docs/ncg_in_jsa_perceptual.md:

1. build distortion-space proposal scores from ``d(x, h)`` gradients;
2. optionally build prior-space proposal scores from ``f(h)`` gradients;
3. sum scores in probability space and apply MH correction on the full joint model.

The stage behaviour is controlled by ``joint_model`` itself. When stage weights disable the prior
term (e.g. stage-1), prior scores are automatically skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from src.base.base_sampler import BaseSampler
from src.utils.codebook_utils import compute_category_weights


@dataclass
class _PriorProposalState:
    """ Intermediate state for computing prior-based proposal scores in NCGSampler.
    
    Args:
        grad_input: Gradient of the prior energy w.r.t. input embeddings, shape [B, T, D_in].
        grad_output: Gradient of the prior energy w.r.t. output embeddings, shape [B, T, D_out].
        input_table: Input embedding table from the prior model, shape [V_in, D_in].
        output_table: Output embedding table from the prior model, shape [V_out, D_out].
        num_categories: Number of categories for each latent variable (for multi-variable cases).
        category_weights: Precomputed weights for encoding multi-dimensional categorical variables.
        weight_tying: Whether the prior model uses weight tying between input and output embeddings.
    """
    
    grad_input: Tensor
    grad_output: Tensor
    input_table: Tensor
    output_table: Tensor
    num_categories: Sequence[int]
    category_weights: Tensor
    weight_tying: bool


class NCGSampler(BaseSampler):
    """Norm constrained local proposal for latent token MCMC."""
    
    def __init__(
        self,
        joint_model: nn.Module,
        proposal_model: Optional[nn.Module] = None, # Not used in this implementation but included for API consistency/future extensions
        alpha: float = 1.0,
        p_norm: float = 2.0,
        num_sweeps: int = 1,
        num_sites: int = 1, # Block size: number of positions updated per MH step. Smaller blocks give higher acceptance rates.
        temperature: float = 1.0,
        include_current_token: bool = True,
        alpha_prior_input: Optional[float] = None,
        alpha_prior_output: Optional[float] = None,
        include_prior_in_proposal: bool = True,
    ):
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.alpha = float(alpha)
        self.p_norm = float(p_norm)
        self.num_sweeps = int(num_sweeps)
        self.num_sites = int(num_sites)
        self.temperature = float(temperature)
        self.include_current_token = bool(include_current_token)
        self.alpha_prior_input = (
            float(alpha) if alpha_prior_input is None else float(alpha_prior_input)
        )
        self.alpha_prior_output = (
            float(alpha) if alpha_prior_output is None else float(alpha_prior_output)
        )
        self.include_prior_in_proposal = bool(include_prior_in_proposal)

    def to(self, device: torch.device):
        return self

    def _prepare_h(self, h: Tensor) -> Tensor:
        """Normalize latent shape to ``[B, ..., num_latent_vars]``."""
        if h.dim() == 5 and h.shape[1] == 1: # [B, 1, H, W, num_latent_vars]
            return h[:, 0].clone() # [B, H, W, num_latent_vars] treated as token indices
        if h.dim() < 3:
            raise ValueError(
                f"NCGSampler expects h with spatial dimensions, got shape {tuple(h.shape)}."
            )
        return h.clone()

    def _get_decoder_and_embeddings(self)-> Tuple[nn.Module, nn.Module]:
        distortion_model = getattr(self.joint_model, "distortion_model", None)
        if distortion_model is None:
            raise AttributeError("joint_model must expose `distortion_model` for NCGSampler.")
        decoder = getattr(distortion_model, "decoder", None)
        if decoder is None or not hasattr(decoder, "embeddings"):
            raise AttributeError(
                "The distortion decoder must expose `embeddings` so that NCG proposals can be computed."
            )
        return distortion_model, decoder

    def _resolve_energy_scales(self, x: Tensor)-> Tuple[Tensor, Tensor]:
        """Resolve distortion and prior energy scales, accounting for any temperature or stage weighting.
        
        Returns:
            distortion_scale: Scalar tensor to scale the distortion energy component in the proposal distribution.
                distortion_scale = distortion_weight / temperature, where distortion_weight is from joint_model._component_weights() if available, else 1.0.
            prior_scale: Scalar tensor to scale the prior energy component in the proposal distribution.
                prior_scale = prior_weight, where prior_weight is from joint_model._component_weights() if available, else 1.0.
        """
        distortion_weight = 1.0
        prior_weight = 1.0
        if hasattr(self.joint_model, "_component_weights"):
            try:
                distortion_weight, prior_weight = self.joint_model._component_weights()
            except Exception:
                pass

        temperature = getattr(self.joint_model, "distortion_temperature", None)
        if temperature is None:
            temperature_tensor = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        elif torch.is_tensor(temperature):
            temperature_tensor = temperature.to(device=x.device, dtype=x.dtype)
        else:
            temperature_tensor = torch.tensor(
                float(temperature), device=x.device, dtype=x.dtype
            )

        distortion_scale = (
            torch.tensor(float(distortion_weight), device=x.device, dtype=x.dtype)
            / temperature_tensor.clamp_min(1e-8)
        )
        prior_scale = torch.tensor(float(prior_weight), device=x.device, dtype=x.dtype)
        return distortion_scale, prior_scale

    def _distortion_embedding_gradients(self, x: Tensor, h: Tensor, distortion_scale: Tensor) -> Tensor:
        """Gradient of stage-aware distortion energy w.r.t. decoder embeddings.
        
        nabla_emb = dE/d_emb = dE/ddistortion * ddistortion/dx_hat * dx_hat/d_emb
        """
        distortion_model, _ = self._get_decoder_and_embeddings()
        embedded_h = distortion_model.embed_latent(h).detach().requires_grad_(True)
        x_hat = distortion_model.decode_from_embedded_latent(embedded_h)
        out = distortion_model.distortion_from_reconstruction(x, x_hat)
        energy = (distortion_scale * out.distortion).sum()
        grad = torch.autograd.grad(energy, embedded_h, only_inputs=True)[0]
        return grad

    def _proposal_scores(
        self,
        grad_site: Tensor,
        current_embed: Tensor,
        candidate_embed: Tensor,
        alpha: float,
    ) -> Tensor:
        """Unnormalized proposal score for candidate embeddings.
        
        score = -0.5 * <grad_site, candidate_embed - current_embed> - 0.5 * alpha * ||candidate_embed - current_embed||_p^p
        """
        diff = candidate_embed - current_embed.unsqueeze(1)
        linear_term = -0.5 * torch.sum(grad_site.unsqueeze(1) * diff, dim=-1)
        norm_term = -0.5 / max(alpha, 1e-8) * torch.sum(diff.abs().pow(self.p_norm), dim=-1)
        scores = linear_term + norm_term
        return scores

    def _flatten_index(self, h: Tensor, latent_var: int):
        """Flatten the specified latent variable across spatial dimensions to get current token indices for masking.
        
        h is expected to have shape [B, ..., num_latent_vars], where ... are spatial dimensions. 
        This method extracts the indices for the specified latent variable and flattens the spatial dimensions, 
        resulting in shape [B, num_tokens]."""
        current_tokens = h[..., latent_var].long()
        return current_tokens.reshape(h.shape[0], -1)

    def _split_gradients(self, grad: Tensor, decoder) -> List[Tensor]:
        dims = list(getattr(decoder, "embedding_dims", []))
        if not dims:
            raise AttributeError(
                "Decoder must define `embedding_dims` for multi-variable NCG proposals."
            )
        return list(torch.split(grad, dims, dim=-1))

    def _mask_current_token(self, scores: Tensor, current_idx: Tensor) -> Tensor:
        if self.include_current_token or scores.shape[1] <= 1:
            # If including current token in proposal or only one token in vocab, no masking needed.
            return scores
        masked = scores.clone()
        masked.scatter_(1, current_idx.unsqueeze(1), float("-inf"))
        all_invalid = torch.isinf(masked).all(dim=1)
        if all_invalid.any():
            masked[all_invalid] = scores[all_invalid]
        return masked

    def _scores_to_probs(self, scores: Tensor) -> Tensor:
        scaled = scores / max(self.temperature, 1e-6)
        probs = F.softmax(scaled, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs

    def _get_prior_model(self):
        """ Helper to get the prior model from the joint model, with checks for required methods."""
        prior_model = getattr(self.joint_model, "prior_model", None)
        if prior_model is None:
            return None
        required = [
            "tokens_from_latent",
            "prepare_teacher_forcing",
            "get_input_embedding_weight",
            "get_output_embedding_weight",
            "forward_logits_from_input_embeddings",
        ]
        if not all(hasattr(prior_model, key) for key in required):
            return None
        return prior_model

    @staticmethod
    def _encode_multidim_tokens(tokens: Tensor, weights: Tensor) -> Tensor:
        """Encode multi-dimensional tokens into a single dimension using learned weights.
        
        Args:
            tokens: Tensor of shape [B, ..., D] where D is the number of dimensions.
            weights: Tensor of shape [D] containing learned weights.
        Returns:
            Tensor of shape [B, ...] containing the encoded tokens.
            
        """
        view_shape = [1] * (tokens.dim() - 1) + [weights.numel()] # reshape weights to broadcast across all but last dimension of tokens
        return (tokens.long() * weights.view(*view_shape)).sum(dim=-1).long()

    def _build_prior_state(
        self,
        h: Tensor,
        prior_scale: Tensor,
    ) -> Optional[_PriorProposalState]:
        if not self.include_prior_in_proposal:
            return None
        if float(prior_scale.item()) == 0.0:
            return None

        prior_model = self._get_prior_model()
        if prior_model is None:
            return None

        tokens, _ = prior_model.tokens_from_latent(h) # tokens shape [B, T, D] where D is num_latent_vars
        inputs, targets = prior_model.prepare_teacher_forcing(tokens)

        in_table_full = prior_model.get_input_embedding_weight() # [V_in, D_in]
        out_table_full = prior_model.get_output_embedding_weight() # [V_out, D_out]

        # in_table size and out_table size should both be >= vocab_size, 
        # where vocab_size is the max token index + 1. This ensures all tokens have valid embeddings.
        vocab_size = int(getattr(prior_model, "vocab_size", out_table_full.shape[0]))
        if in_table_full.shape[0] <= int(getattr(prior_model, "sos_token", vocab_size)):
            return None
        if out_table_full.shape[0] < vocab_size:
            return None

        input_embeddings = F.embedding(inputs.long(), in_table_full).detach().requires_grad_(True)
        logits, _, hidden = prior_model.forward_logits_from_input_embeddings(
            input_embeddings,
            return_hidden=True,
        ) # logits shape [B, T, V_out], hidden shape [B, T, D_hidden]
        if logits.shape[:2] != targets.shape:
            # If the prior model changes the sequence length (e.g. via pooling), 
            # we cannot reliably compute token-level proposal scores, so we skip the prior component.
            return None

        # 使用内置的交叉熵直接计算 sum([-log P(y|x)])，更加简洁和高效
        energy_input = prior_scale * F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), 
            targets.reshape(-1), 
            reduction="sum"
        )
        
        grad_shifted = torch.autograd.grad(
            energy_input,
            input_embeddings,
            retain_graph=True,
            only_inputs=True,
        )[0] # shape [B, T, D_in]

        # The proposal scores will be computed for candidate tokens at each position. 
        # The gradient from the prior energy w.r.t. the input embedding at each position indicates how changing the token at that position would affect the prior energy.
        # The reason for the "shift" in grad_shifted is that the input embedding at position i affects the prior energy through the output logits at position i, 
        # which in turn depend on the hidden state at position i.
        grad_input = torch.zeros_like(grad_shifted)
        if grad_shifted.shape[1] > 1:
            grad_input[:, :-1, :] = grad_shifted[:, 1:, :]

        #! Important: must be detached to avoid backprop through the prior model when computing proposal scores, which would violate the MCMC framework.
        out_table = out_table_full[:vocab_size].detach() # we don't need the SOS token
        hidden = hidden.detach()
        targets = targets.long()

        # Direct application of chain rule gives us the gradient of the prior energy w.r.t. the output embeddings at each position, 
        # which is what we need to compute proposal scores for candidate tokens at that position.
        # d(-log(p_y)) / d(E_y) = (p_y - 1) * h
        logits_detached = torch.einsum("btd,vd->btv", hidden, out_table)
        probs = F.softmax(logits_detached, dim=-1)
        target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        grad_output = prior_scale * (target_probs - 1.0).unsqueeze(-1) * hidden

        num_categories = list(getattr(prior_model, "num_categories", []))
        if len(num_categories) == 0:
            num_categories = [vocab_size]
        if len(num_categories) != h.shape[-1]:
            return None

        category_weights = compute_category_weights(
            num_categories,
            device=h.device,
            dtype=torch.long,
        )

        return _PriorProposalState(
            grad_input=grad_input,
            grad_output=grad_output,
            input_table=in_table_full[:vocab_size].detach(),
            output_table=out_table,
            num_categories=num_categories,
            category_weights=category_weights,
            weight_tying=bool(getattr(prior_model, "uses_weight_tying", False)),
        )

    def _prior_site_scores(
        self,
        prior_state: _PriorProposalState,
        flat_tokens: Tensor,
        position: int,
        latent_var: int,
    ) -> Optional[Tensor]:
        K = int(prior_state.num_categories[latent_var])
        current_site = flat_tokens[:, position, :].long() # shape [B, num_latent_vars]
        if len(prior_state.num_categories) == 1:
            candidate_ids = torch.arange(K, device=flat_tokens.device).view(1, K)
            candidate_ids = candidate_ids.expand(flat_tokens.shape[0], -1) # shape [B, K]
            current_ids = current_site[:, 0] # shape [B]
        else:
            candidate_grid = current_site.unsqueeze(1).repeat(1, K, 1) # shape [B, K, num_latent_vars]
            candidate_values = torch.arange(K, device=flat_tokens.device).view(1, K)
            candidate_grid[:, :, latent_var] = candidate_values
            candidate_ids = self._encode_multidim_tokens(
                candidate_grid,
                prior_state.category_weights,
            )
            current_ids = self._encode_multidim_tokens(
                current_site,
                prior_state.category_weights,
            )

        if candidate_ids.max().item() >= prior_state.input_table.shape[0]:
            return None

        grad_in = prior_state.grad_input[:, position, :]
        grad_out = prior_state.grad_output[:, position, :]

        cur_in = prior_state.input_table[current_ids]
        cand_in = prior_state.input_table[candidate_ids]

        if prior_state.weight_tying:
            # Input and output embedding tables are identical (shared weights).
            # The total gradient of f_psi w.r.t. the shared embedding at position i
            # is grad_in[i] + grad_out[i].  We apply a single norm constraint
            # (not two), so only one call to _proposal_scores is needed.
            combined_grad = grad_in + grad_out
            return self._proposal_scores(
                grad_site=combined_grad,
                current_embed=cur_in,
                candidate_embed=cand_in,
                alpha=self.alpha_prior_input,
            )

        # Without weight tying the two embedding spaces are distinct.
        # Compute scores independently in each space and sum them,
        # following docs/ncg_in_jsa_perceptual.md 思路2.
        score_in = self._proposal_scores(
            grad_site=grad_in,
            current_embed=cur_in,
            candidate_embed=cand_in,
            alpha=self.alpha_prior_input,
        )
        cur_out = prior_state.output_table[current_ids]
        cand_out = prior_state.output_table[candidate_ids]
        score_out = self._proposal_scores(
            grad_site=grad_out,
            current_embed=cur_out,
            candidate_embed=cand_out,
            alpha=self.alpha_prior_output,
        )
        return score_in + score_out

    @staticmethod
    def _as_batch_vector(t: Tensor) -> Tensor:
        """Ensure the input tensor is of shape [B] for batch processing, flattening any additional dimensions."""
        if t.dim() == 1:
            return t
        return t.view(t.shape[0], -1).mean(dim=1)

    def _compute_site_probs(
        self,
        latent_var: int,
        pos: int,
        flat_tokens: Tensor,
        embedding_weight: Tensor,
        grad_grid: Tensor,
        prior_state: Optional[_PriorProposalState],
    ) -> Tuple[Tensor, Tensor]:
        """Compute proposal probabilities for a specific spatial site and latent variable."""
        batch_size = flat_tokens.shape[0]
        cur_idx = flat_tokens[:, pos, latent_var].long()
        cur_embed = embedding_weight[cur_idx]
        grad_site = grad_grid[:, pos]

        score_total = self._proposal_scores(
            grad_site=grad_site,
            current_embed=cur_embed,
            candidate_embed=embedding_weight.unsqueeze(0).expand(batch_size, -1, -1),
            alpha=self.alpha,
        )

        if prior_state is not None:
            prior_scores = self._prior_site_scores(
                prior_state=prior_state,
                flat_tokens=flat_tokens,
                position=pos,
                latent_var=latent_var,
            )
            if prior_scores is not None and prior_scores.shape == score_total.shape:
                score_total = score_total + prior_scores

        score_total = self._mask_current_token(score_total, cur_idx)
        probs = self._scores_to_probs(score_total)
        return probs, cur_idx

    def _propose(self, x: Tensor, h: Tensor, position_ids: Optional[Tensor] = None):
        distortion_model, decoder = self._get_decoder_and_embeddings()
        distortion_scale, prior_scale = self._resolve_energy_scales(x)

        grad = self._distortion_embedding_gradients(x, h, distortion_scale)
        grad_splits = self._split_gradients(grad, decoder)
        prior_state = self._build_prior_state(h, prior_scale)

        proposal_log_prob = torch.zeros(h.shape[0], device=h.device)
        h_new = h.clone().long()
        selected = []

        batch_size = h.shape[0]
        num_latent_vars = h.shape[-1]
        num_positions = h[..., 0].numel() // h.shape[0]
        num_sites = min(self.num_sites, num_positions)
        if num_sites <= 0:
            return h_new.float(), proposal_log_prob, selected

        if position_ids is None:
            position_ids = torch.randperm(num_positions, device=h.device)[:num_sites]

        flat_tokens = h_new.view(batch_size, num_positions, num_latent_vars)

        for latent_var, (embedding, grad_chunk) in enumerate(zip(decoder.embeddings, grad_splits)):
            token_grid = flat_tokens[:, :, latent_var]
            grad_grid = grad_chunk.reshape(h.shape[0], num_positions, -1)
            embedding_weight = embedding.weight.detach()

            for pos in position_ids.tolist():
                probs, cur_idx = self._compute_site_probs(
                    latent_var, pos, flat_tokens, embedding_weight, grad_grid, prior_state
                )

                sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                proposal_log_prob = proposal_log_prob + torch.log(
                    probs.gather(1, sampled_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
                )
                token_grid[:, pos] = sampled_idx
                selected.append((latent_var, pos))

            flat_tokens[:, :, latent_var] = token_grid

        h_new = flat_tokens.view_as(h_new)

        return h_new.float(), proposal_log_prob, selected

    def _reverse_log_prob(self, x: Tensor, h_from: Tensor, h_to: Tensor, selected):
        distortion_model, decoder = self._get_decoder_and_embeddings()
        distortion_scale, prior_scale = self._resolve_energy_scales(x)

        grad = self._distortion_embedding_gradients(x, h_to, distortion_scale)
        grad_splits = self._split_gradients(grad, decoder)
        prior_state = self._build_prior_state(h_to, prior_scale)

        num_positions = h_to[..., 0].numel() // h_to.shape[0]
        num_latent_vars = h_to.shape[-1]
        reverse_log_prob = torch.zeros(h_to.shape[0], device=h_to.device)
        batch_size = h_to.shape[0]

        # Mirror the forward proposal exactly: start from h_to and progressively revert
        # each site to h_from in the same order as the forward loop.
        #
        # This matters for multi-latent-var cases where _prior_site_scores reads
        # flat_tokens[:, pos, :] (all latent vars at a position) to build candidate_ids.
        # In the forward, when processing (lv=L, pos=P), lv < L are already at their
        # new values.  The reverse must mirror this: when evaluating the reverse
        # probability for (lv=L, pos=P), latent vars < L should already have been
        # reverted (back to h_from), exactly matching the partially-updated state
        # used in the forward.
        reverse_flat = h_to.long().view(batch_size, num_positions, num_latent_vars).clone()
        h_from_flat = h_from.long().view(batch_size, num_positions, num_latent_vars)

        for latent_var, pos in selected:
            embedding = decoder.embeddings[latent_var]
            embedding_weight = embedding.weight.detach()
            grad_grid = grad_splits[latent_var].reshape(batch_size, num_positions, -1)

            old_idx = h_from_flat[:, pos, latent_var].long()

            probs, _ = self._compute_site_probs(
                latent_var, pos, reverse_flat, embedding_weight, grad_grid, prior_state
            )

            reverse_log_prob = reverse_log_prob + torch.log(
                probs.gather(1, old_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            )

            # Revert this site so subsequent steps see the correct partially-restored state.
            reverse_flat[:, pos, latent_var] = old_idx

        return reverse_log_prob

    def step(self, x: Tensor, h: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        h_old = self._prepare_h(h)
        h_new, log_q_forward, selected = self._propose(x, h_old, position_ids=position_ids)
        if len(selected) == 0:
            return h_old.float()

        log_q_reverse = self._reverse_log_prob(x, h_old, h_new, selected)

        log_p_old = self._as_batch_vector(self.joint_model.log_joint_prob(x, h_old))
        log_p_new = self._as_batch_vector(self.joint_model.log_joint_prob(x, h_new))
        log_accept = (log_p_new + log_q_reverse) - (log_p_old + log_q_forward)
        
        # 使用对数空间的均匀分布判定接受率，免去 exp 和 clamp 操作，防止极小概率下的截断错误
        log_u = torch.log(torch.rand_like(log_accept).clamp_min(1e-8))
        accept = log_u < log_accept
        
        accept_view = accept.view(-1, *([1] * (h_old.dim() - 1)))
        h_next = torch.where(accept_view, h_new, h_old)
        return h_next.float()

    def sample(self, x: Tensor, h_init: Tensor, num_sweeps: Optional[int] = None, return_all: bool = False):
        """Sample by performing full sweeps over all positions.

        Each sweep visits every position exactly once in a random order, processing
        them in blocks of ``num_sites``.  A separate MH accept/reject step is applied
        to each block, keeping acceptance rates high while still covering the whole
        sequence per sweep.

        Args:
            x: Observed data.
            h_init: Initial latent token sequence.
            num_sweeps: Number of full sweeps. Defaults to ``self.num_sweeps``.
            return_all: If True, return the sequence after every sweep stacked as
                ``[B, num_sweeps, ...]``; otherwise return only the final sample.
        """
        sweeps = self.num_sweeps if num_sweeps is None else int(num_sweeps)
        h = self._prepare_h(h_init)
        num_positions = h[..., 0].numel() // h.shape[0]
        block_size = max(1, min(self.num_sites, num_positions))

        traj = []
        for _ in range(sweeps):
            perm = torch.randperm(num_positions, device=h.device)
            for start in range(0, num_positions, block_size):
                block_ids = perm[start : start + block_size]
                h = self.step(x, h, position_ids=block_ids)
            if return_all:
                traj.append(h)

        if return_all:
            return torch.stack(traj, dim=1)
        return h


__all__ = ["NCGSampler"]
