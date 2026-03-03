# src/modules/gpts/mingpt.py
"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier

There is something different from the minGPT/nanoGPT implementation in that the attention mask is registered as a buffer, which allows it to be used on the GPU without needing to be re-created each time. This is a minor change that should not affect the functionality of the model.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[: config.n_unmasked, : config.n_unmasked] = 1
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present  # TODO: check that this does not break anything


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present:
            assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=12,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        n_unmasked=0,
    ):
        super().__init__()
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
        )
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_with_past(
        self, idx, embeddings=None, targets=None, past=None, past_length=None
    ):
        """Forward using KV cache. Only for inference at the moment."""
        # inference only
        assert not self.training
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)  # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [
                self.config.n_layer,
                2,
                idx.shape[0],
                self.config.n_head,
                past_length,
                self.config.n_embd // self.config.n_head,
            ]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[
                :, past_length, :
            ]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(
                x,
                layer_past=past[i, ...] if past is not None else None,
                return_present=True,
            )
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return (
            logits,
            loss,
            torch.stack(presents),
        )  # _, _, n_layer, 2, b, nh, 1, dim_head

    @torch.no_grad()
    def sample(
        self,
        idx,
        steps,
        temperature=1.0,
        top_k=None,
        top_p=None,
        callback=lambda k: None,
    ):
        """Take a conditioning sequence of indices in idx (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        block_size = self.get_block_size()
        self.eval()
        for k in range(steps):
            callback(k)

            # Limit the context to block_size tokens, as the model cannot attend to more than that
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[
                :, -1, :
            ]  # pluck the logits at the final step for the next token prediction

            if temperature > 0:
                logits = logits / temperature
                if top_k is not None or top_p is not None:
                    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(logits, k=1, dim=-1)

            # append to the sequence and continue
            idx = torch.cat((idx, ix), dim=1)

        return idx

    @torch.no_grad()
    def sample_with_past(
        self,
        idx,
        steps,
        temperature=1.0,
        top_k=None,
        top_p=None,
        callback=lambda k: None,
    ):
        # idx is the conditioning sequence of indices, of shape (b, t)
        # current_input is the input for the current step, initialized as idx and updated to the newly sampled token at each step
        sample = idx
        current_input = idx
        cond_len = idx.shape[1]

        # past is a list of length n_layer, each element is a tuple of (key, value) with shape (b, nh, T_past, hs)
        # List[Tensor[layer, 2, b, nh, T_past, hs]]
        past = None
        for n in range(steps):
            callback(n)
            logits, _, present = self.forward_with_past(
                current_input,
                past=past,
                past_length=sample.shape[1] - current_input.shape[1],
            )

            # Update KV cache
            if past is None:
                past = [present]
            else:
                past.append(present)

            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k is not None or top_p is not None:
                    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(logits, k=1, dim=-1)

            sample = torch.cat((sample, idx_next), dim=1)
            current_input = idx_next  # for the next step's input

        del past  # free memory
        sample = sample[:, cond_len:]  # cut conditioning off
        return sample


class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


#### sampling utils
# New version: using `LogitsWarper` from transformers
from transformers import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
):
    warpers = []
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k=top_k))
    if top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p))

    for warper in warpers:
        logits = warper(None, logits)

    return logits


""" Functions below can be deleted.

- top_k_logits: old version of top_k_top_p_filtering, only supports top_k.
- sample/sample_with_past: they are moved to the `GPT` class as methods.
- _top_k_top_p_filtering: old version of top_k_top_p_filtering, have been replaced by the new version using `LogitsWarper` from transformers.
"""


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


# Old version of top_k_top_p_filtering
def _top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


@torch.no_grad()
def sample_with_past(
    x,
    model,
    steps,
    temperature=1.0,
    sample_logits=True,
    top_k=None,
    top_p=None,
    callback=None,
):
    # x is conditioning
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(
            x, past=past, past_length=(n + cond_len - 1)
        )
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample
