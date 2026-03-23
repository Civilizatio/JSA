from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.codebook_utils import encode_multidim_to_index


class UniformPriorEnergy(nn.Module):
    """Constant prior energy. Useful for stage 1 pretraining or ablations."""

    def __init__(self, num_categories=None):
        super().__init__()
        self.num_categories = num_categories

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns a constant energy for each sample in the batch.

        For uniform prior, the energy is constant across the latent space, namely f_psi(h) = C for all h, so the value does not matter.
        Theorietically, C should be -log(1/|H|) where |H| is the size of the latent space, but it could be any constant since it does not affect the gradients.
        """
        return torch.zeros(h.shape[0], device=h.device, dtype=torch.float32)


class GPTPriorEnergy(nn.Module):
    """Autoregressive prior energy f_psi(h) = -log p_psi(h).

    The latent tensor ``h`` is flattened to a token sequence and then scored with
    a GPT-style causal language model.
    """

    def __init__(
        self,
        transformer: nn.Module,
        num_categories: Sequence[int] | int,
        bos_token: int | None = None,
    ):
        super().__init__()
        if isinstance(num_categories, int):
            self.num_categories = [num_categories]
        else:
            self.num_categories = list(num_categories)
        self.transformer = transformer
        self.codebook_size = 1
        for k in self.num_categories:
            self.codebook_size *= int(k)

        # By default, use the codebook size as the bos token, which is one index after the last valid codebook token.
        self.bos_token = self.codebook_size if bos_token is None else int(bos_token)

        if hasattr(transformer, "config"):
            vocab_size = transformer.config.vocab_size
        elif hasattr(transformer, "tok_emb"):
            vocab_size = transformer.tok_emb.num_embeddings
        else:
            vocab_size = None

        required_vocab = max(self.codebook_size, self.bos_token + 1)
        if vocab_size is not None and vocab_size < required_vocab:
            raise ValueError(
                f"Transformer vocab_size={vocab_size} is smaller than required {required_vocab}."
            )

    def _flatten_latents(self, h: torch.Tensor) -> torch.Tensor:
        """Flattens the latent tensor h to a 2D token sequence of shape [batch, seq_len].

        The last dimension of h is treated as the categorical dimension, and is converted to token indices using the codebook sizes. 
        For example, if num_categories=[4, 8], then each latent variable is a pair of integers (c1, c2) where c1 in [0, 3] and c2 in [0, 7], 
        and is converted to a single token index c = c1 * 8 + c2 in [0, 31]. 
        If num_categories=[16], then each latent variable is a single integer in [0, 15] and is already a token index.
        
        Arguments:
            h: Latent tensor of shape [batch, ... , num_latent_vars], where num_latent_vars is the size of the last dimension and represents the number of categorical latent variables per
                spatial location. The other dimensions can be arbitrary (e.g. height and width for images).
        Returns:
            tokens: LongTensor of shape [batch, seq_len] where seq_len is the product of the spatial dimensions of h, and each value is a token index corresponding to the categorical latent variables
                at that spatial location.
        """
        if h.dim() < 2:
            raise ValueError(
                f"Expected latent tensor with at least 2 dims, got {h.shape}"
            )

        if h.dim() == 2:
            if len(self.num_categories) != 1:
                raise ValueError(
                    "2D latent tensors are only supported when num_latent_vars == 1."
                )
            return h.long()

        num_latent_vars = h.shape[-1]
        h_flat = h.reshape(h.shape[0], -1, num_latent_vars).long() # [B, D1, D2, ..., num_latent_vars] -> [B, seq_len, num_latent_vars]
        if num_latent_vars == 1:
            tokens = h_flat.squeeze(-1)
        else:
            tokens = encode_multidim_to_index(
                h_flat.reshape(-1, num_latent_vars),
                self.num_categories,
            ).view(h.shape[0], -1)
        return tokens.long() # [B, seq_len]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        tokens = self._flatten_latents(h)
        batch, seq_len = tokens.shape
        bos = torch.full(
            (batch, 1),
            self.bos_token,
            device=tokens.device,
            dtype=torch.long,
        )
        input_tokens = torch.cat([bos, tokens[:, :-1]], dim=1)
        logits, _ = self.transformer(input_tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
            reduction="none",
        )
        return loss.view(batch, seq_len).sum(dim=1)
