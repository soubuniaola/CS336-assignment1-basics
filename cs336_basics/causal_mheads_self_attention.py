from torch import nn
from einops import rearrange, einsum
import torch

from cs336_basics.functional_utils import scaled_dot_product_attention
from cs336_basics.rope import Rope


# causal multi-head self-attention
class CausalMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Rope| None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = nn.Parameter(torch.randn(num_heads * d_model // num_heads, d_model))
        self.k_proj = nn.Parameter(torch.randn(num_heads * d_model // num_heads, d_model))
        self.v_proj = nn.Parameter(torch.randn(num_heads * d_model // num_heads, d_model))
        self.o_proj = nn.Parameter(torch.randn(num_heads * d_model // num_heads, d_model))

        self.rope = rope

    def forward(self, in_feature: torch.Tensor) -> torch.Tensor:
        q_proj = rearrange(self.q_proj, "(num_heads d) d_model -> num_heads d d_model", num_heads=self.num_heads)
        k_proj = rearrange(self.k_proj, "(num_heads d) d_model -> num_heads d d_model", num_heads=self.num_heads)
        v_proj = rearrange(self.v_proj, "(num_heads d) d_model -> num_heads d d_model", num_heads=self.num_heads)
        o_proj = self.o_proj

        Q = einsum(q_proj, in_feature, "num_heads sub_queries d_model, ... seq_len d_model -> ... num_heads seq_len sub_queries")
        K = einsum(k_proj, in_feature, "num_heads sub_keys d_model, ... seq_len d_model -> ... num_heads seq_len sub_keys")
        V = einsum(v_proj, in_feature, "num_heads sub_values d_model, ... seq_len d_model -> ... num_heads seq_len sub_values")

        if self.rope:
            Q = self.rope.forward(Q,self.rope.token_position)
            K = self.rope.forward(K,self.rope.token_position)

        seq_len = in_feature.size(-2)  # the second-to-last dimension
        batch_shape = in_feature.shape[:-2]  # everything except (seq_len, d_in)

        # make base mask (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  # causal mask

        # now expand it to (..., seq_len, seq_len)
        mask = mask.expand(*batch_shape,self.num_heads, seq_len, seq_len)

        # batched_attention: (... num_heads sub_queries seq_len) 4 16 12
        batched_attention = scaled_dot_product_attention(Q, K, V, mask=mask)

        batched_attention = rearrange(batched_attention, "... num_heads seq_len sub_dim -> ... (num_heads sub_dim) seq_len")
        multihead_attention = einsum(o_proj, batched_attention, "d_model d_v, ... d_v seq_len -> ... seq_len d_model")
        return multihead_attention

