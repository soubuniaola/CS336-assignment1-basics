import einx
from torch import nn
from einops import rearrange, einsum
import torch

from cs336_basics.function_utils import scaled_dot_product_attention
from cs336_basics.linear_module import Linear
from cs336_basics.rope import Rope


# causal multi-head self-attention
class CausalMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Rope | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        d_head = d_model // num_heads
        d_v = d_head * num_heads

        self.q_proj = Linear(d_model, d_head * num_heads)
        self.k_proj = Linear(d_model, d_head * num_heads)
        self.v_proj = Linear(d_model, d_head * num_heads)
        self.output_proj = Linear(d_model, d_v)

        self.rope = rope

    def forward(self, in_feature: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *b, sequence_length, d_model = in_feature.size()
        Q = self.q_proj(in_feature)
        K = self.k_proj(in_feature)
        V = self.v_proj(in_feature)

        Q = rearrange(Q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)

        if self.rope:
            if token_positions is None:
                token_positions = einx.rearrange("seq -> b... seq",
                                                 torch.arange(sequence_length),
                                                 b=[1] * len(b))

            # Duplicate token positions for each head
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

            Q = self.rope.forward(Q,token_positions)
            K = self.rope.forward(K,token_positions)

        seq_len = in_feature.size(-2)  # the second-to-last dimension
        batch_shape = in_feature.shape[:-2]  # everything except (seq_len, d_in)

        # make base mask (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  # causal mask

        # now expand it to (..., seq_len, seq_len)
        mask = mask.expand(*batch_shape,self.num_heads, seq_len, seq_len)

        # batched_attention: (... num_heads sub_queries seq_len) 4 16 12
        batched_attention = scaled_dot_product_attention(Q, K, V, mask=mask)

        batched_attention = rearrange(batched_attention, "... num_heads seq_len sub_dim -> ... seq_len (num_heads sub_dim)")
        #multihead_attention = einsum(o_proj, batched_attention, "d_model d_v, ... d_v seq_len -> ... seq_len d_model")
        attn_out = self.output_proj(batched_attention)
        return attn_out

