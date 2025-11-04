import torch
from torch import nn

from cs336_basics.causal_mheads_self_attention import CausalMHA
from cs336_basics.rmsnorm_module import RMSNorm
from cs336_basics.rope import Rope
from cs336_basics.swiglu import SwiGlu


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: Rope
    ):
        super().__init__()
        self.attn = CausalMHA(
            d_model = d_model,
            num_heads = num_heads,
            rope = positional_encoder)
        self.ffn = SwiGlu(
            d_model = d_model,
            d_ff = d_ff,
        )
        self.ln1 = RMSNorm(d_model = d_model)
        self.ln2 = RMSNorm(d_model = d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + attn

        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = x_ffn + attn_sublayer_output
        return ffn_sublayer_output


