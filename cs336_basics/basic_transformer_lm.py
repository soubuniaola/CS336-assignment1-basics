from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat
from cs336_basics.embedding import Embedding
from cs336_basics.function_utils import softmax
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import Rope
from cs336_basics.transformer_block import TransformerBlock
from torch import Tensor
from jaxtyping import Float, Bool, Int
import logging

logger = logging.getLogger(__name__)

class BasicTransformerLM(nn.Module):
    """ A Transformer Language Model
    Args:
        vocab_size: int,
            - vocabulary size for the model
        context_length: int,
            - maximum context length
        d_model: int,
            - token / embedding dimension
        num_layers: int,
            - number of transformer blocks
        d_ff: int,
            - dimension for feed-forward layers
                (normally 4*d_model; 8/3*d_model for multi-head attention)
        num_heads: int,
            - number of attention heads
                ('d_model' must be evenly divisible by num_heads)
        rope_theta: int,
            - THETA for ROPE positional encoding

    Returns:
        FloatTensor of shape (batch_size, seq_len, vocab_size) with
        the predicted unnormalized next-word distribution for each token
    """

    def __init__(self, vocab_size, context_length, d_model, num_layers, d_ff, num_heads, rope_theta):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }

        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = Rope(
            theta=rope_theta,
            max_seq_len=context_length,
            dim=d_head
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count(default), with lm_head being subtracted.
        """
        n_para = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_para -= self.lm_head.weight.numel()

        return n_para

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        """
        Args:
            x: Input token IDs for the language model

        Returns:
            FloatTensor
            of shape (batch_size, seq_len, vocab_size) with
            the predicted unnormalized next-word distribution
            for each token
        """

        _, seq_len = x.shape
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        eos_id: int | None = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        generated = torch.empty([1,1])
        origin_seq_len = x.size(-1)
        for _ in range(max_new_tokens):
            # only get the last context_length tokens if length of x
            # exceeds the model's context_length
            x = x[:, -self.context_length:] if x.size(1) > self.context_length else x

            # get the next token's logits (..., seq_len, vocab_size)
            logits = self(x)
            next_token_logits = logits[:, -1]
            assert next_token_logits.size() == (x.size(0), self.vocab_size), "token shape: {}".format(logits.size())


            # apply temperature scaling
            temperature_scaled_logits = next_token_logits / temperature
            temperature_scaled_probs = softmax(temperature_scaled_logits, dimension=-1)

            # apply top-p sampling
            sorted_probs, sorted_indices = torch.sort(temperature_scaled_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # mask out tokens with cumulative prob > top_p
            mask = cumulative_probs > top_p

            # ensure at least one token is preserved
            mask[...,1:] = mask[...,:-1].clone()
            mask[...,0] = False

            # fancy indexing
            sorted_probs = sorted_probs[mask]
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # get the next token id
            next_token = torch.multinomial(sorted_probs, 1).unsqueeze(-1)
            next_token = sorted_indices.gather(-1, next_token)

            if eos_id is not None and next_token.item() == eos_id:
                break
            x = torch.cat((x, next_token), dim=-1)
            generated = torch.cat((generated, next_token), dim=-1)

        new_token_ids = x[:,origin_seq_len:]
        return generated[:,1:]

