import torch
from torch import nn, Tensor
from jaxtyping import Float
from einops import einsum

from cs336_basics import functional_utils


class SwiGlu(nn.Module):
    def __init__(self,
    d_model: int,
    d_ff: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.randn(d_ff, d_model))
        self.w2_weight = nn.Parameter(torch.randn(d_model, d_ff))
        self.w3_weight = nn.Parameter(torch.randn(d_ff, d_model,))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        W1_x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3_x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        glu_out = activations.silu(W1_x) * W3_x
        swiglu_out = einsum(self.w2_weight, glu_out, "d_model d_ff, ... d_ff -> ... d_model")
        return swiglu_out