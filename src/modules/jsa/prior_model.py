"""Prior energy modules for energy-based JSA."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.codebook_utils import decode_index_to_multidim, encode_multidim_to_index


@dataclass
class PriorAnalysisOutput:
    """Structured output for autoregressive prior analysis."""
    prior_nll: Tensor
    prior_perplexity: Tensor
    token_entropy: Tensor

class UniformPriorEnergy(nn.Module):
    """A zero-energy prior.

    Theoretically this corresponds to a uniform distribution over the latent space, but in practice it just means we don't apply any prior regularization at all.
    This keeps stage-1 behaviour fully compatible with the original decoder-only JSA setup.
    """

    def __init__(self, num_categories: Optional[Sequence[int]] = None):
        super().__init__()
        self.num_categories = list(num_categories) if num_categories is not None else None

    def forward(self, h: Tensor) -> Tensor:
        """Returns a zero energy for any input tensor h."""
        return torch.zeros(h.shape[0], device=h.device, dtype=torch.float32)

    def get_loss(self, h: Tensor) -> Tensor:
        return self.forward(h).mean()

    def analyze(self, h: Tensor) -> PriorAnalysisOutput:
        zero = torch.tensor(0.0, device=h.device)
        return PriorAnalysisOutput(
            prior_nll=zero,
            prior_perplexity=torch.tensor(1.0, device=h.device),
            token_entropy=zero
        )


class GPTPriorEnergy(nn.Module):
    """Autoregressive token prior interpreted as an energy ``f_psi(h) = -log p_psi(h)``.

    Parameters
    ----------
    net:
        Any autoregressive network that follows the ``minGPT`` convention and returns either
        ``logits`` or ``(logits, aux)`` from ``forward``.
    num_categories:
        Number of categories for each latent variable at one spatial location.
    num_latent_vars:
        Number of categorical variables at one position. When this is greater than one, the
        variables are packed into a single token index via mixed-radix encoding.
    sos_token:
        Start-of-sequence token id. By convention this should be ``vocab_size - 1`` when the
        autoregressive model reserves one extra token for BOS.
    flatten_order:
        Currently only ``"raster"`` is supported.
    """

    def __init__(
        self,
        net: nn.Module,
        num_categories: Sequence[int],
        num_latent_vars: int = 1,
        sos_token: Optional[int] = None,
        flatten_order: str = "raster",
        reduction: str = "mean",
    ):
        super().__init__()
        self.net = net
        self.num_latent_vars = int(num_latent_vars)
        if isinstance(num_categories, int):
            self.num_categories = [num_categories] * self.num_latent_vars
        elif len(num_categories) == 1 and self.num_latent_vars > 1:
            self.num_categories = list(num_categories) * self.num_latent_vars
        else:
            self.num_categories = list(num_categories)
        self.flatten_order = flatten_order
        self.reduction = reduction
        self.vocab_size = math.prod(self.num_categories)
        self.sos_token = self.vocab_size if sos_token is None else int(sos_token)

    def _to_token_sequence(self, h: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
        """Converts a latent tensor h of shape [B, ..., num_latent_vars] into a token sequence of shape [B, T], where T is the number of spatial positions.
        Also returns the original spatial shape for later unflattening.
        """
        if h.dim() < 2:
            raise ValueError(f"Expected h with batch dimension, but received shape {tuple(h.shape)}.")
        if h.shape[-1] != self.num_latent_vars:
            raise ValueError(
                f"The last dimension of h must equal num_latent_vars={self.num_latent_vars}, got shape {tuple(h.shape)}."
            )

        batch_size = h.shape[0]
        spatial_shape = tuple(h.shape[1:-1]) # [..., num_latent_vars]
        h_flat = h.reshape(batch_size, -1, self.num_latent_vars).long() # [B, T, num_latent_vars]

        if self.flatten_order != "raster":
            raise ValueError(f"Unsupported flatten_order: {self.flatten_order!r}")

        if self.num_latent_vars == 1:
            tokens = h_flat.squeeze(-1) # [B, T]
        else:
            tokens = encode_multidim_to_index(
                h_flat.reshape(-1, self.num_latent_vars), self.num_categories
            ).view(batch_size, -1) # from [B*T, num_latent_vars] to [B, T]
        return tokens, spatial_shape

    def _tokens_to_latent(self, tokens: Tensor, spatial_shape: Tuple[int, ...]) -> Tensor:
        """Converts a token sequence of shape [B, T] back into a latent tensor of shape [B, ..., num_latent_vars]."""
        batch_size, seq_len = tokens.shape
        if self.num_latent_vars == 1:
            h = tokens.view(batch_size, *spatial_shape, 1) # [B, ..., 1]
            return h.float()

        decoded = decode_index_to_multidim(tokens.reshape(-1), self.num_categories)
        decoded = decoded.view(batch_size, seq_len, self.num_latent_vars)
        decoded = decoded.view(batch_size, *spatial_shape, self.num_latent_vars)
        return decoded.float()

    def _forward_logits(self, idx: Tensor) -> Tensor:
        """Forwards the token indices through the autoregressive net to get logits of shape [B, T, V], where V is the vocab size."""
        outputs = self.net(idx)
        if isinstance(outputs, tuple):
            # Assume the first element is the logits and ignore any auxiliary outputs.
            logits = outputs[0]
        else:
            logits = outputs
        return logits

    def _prepare_teacher_forcing(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, _ = tokens.shape
        bos = torch.full(
            (batch_size, 1),
            self.sos_token,
            device=tokens.device,
            dtype=tokens.dtype,
        ) # bos token of shape [B, 1]
        inputs = torch.cat([bos, tokens[:, :-1]], dim=1)
        return inputs, tokens

    def forward(self, h: Tensor) -> Tensor:
        tokens, _ = self._to_token_sequence(h)
        inputs, targets = self._prepare_teacher_forcing(tokens)
        logits = self._forward_logits(inputs)
        if logits.shape[:2] != targets.shape:
            raise RuntimeError(
                f"Prior network returned logits of shape {tuple(logits.shape)}, expected [B, T, V] with T={targets.shape[1]}."
            )
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return -token_log_probs.sum(dim=1)

    def get_loss(self, h: Tensor) -> Tensor:
        # The training loop uses a single latent sample in stages 2 and 3. We still support
        # tensors with an explicit sample dimension of shape [B, S, ..., num_latent_vars].
        if h.dim() >= 5:
            batch_size, num_samples = h.shape[0], h.shape[1]
            flat_h = h.reshape(batch_size * num_samples, *h.shape[2:])
            return self.forward(flat_h).mean()
        return self.forward(h).mean()

    @torch.no_grad()
    def analyze(self, h: Tensor) -> PriorAnalysisOutput:
        tokens, _ = self._to_token_sequence(h)
        inputs, targets = self._prepare_teacher_forcing(tokens)
        logits = self._forward_logits(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        token_nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        nll = token_nll.mean()
        perplexity = torch.exp(token_nll.mean())
        return PriorAnalysisOutput(
            prior_nll=nll,
            prior_perplexity=perplexity,
            token_entropy=entropy.mean(),
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        spatial_shape: Sequence[int],
        device=None,
        temperature: float = 1.0,
        top_k=None,
        top_p=None,
    ) -> Tensor:
        device = device or next(self.parameters()).device
        seq_len = int(math.prod(tuple(spatial_shape)))
        bos = torch.full((batch_size, 1), self.sos_token, device=device, dtype=torch.long)

        if hasattr(self.net, "sample"):
            sampled = self.net.sample(
                bos,
                steps=seq_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens = sampled[:, 1:] if sampled.shape[1] == seq_len + 1 else sampled[:, -seq_len:]
        else:
            tokens = bos
            for _ in range(seq_len):
                logits = self._forward_logits(tokens)
                next_logits = logits[:, -1, :]
                if temperature > 0:
                    next_logits = next_logits / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
            tokens = tokens[:, 1:]

        return self._tokens_to_latent(tokens, tuple(spatial_shape))


__all__ = ["GPTPriorEnergy", "UniformPriorEnergy"]
