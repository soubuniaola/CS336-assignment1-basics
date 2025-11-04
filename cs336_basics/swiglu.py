import torch
from torch import nn, Tensor
from jaxtyping import Float
from einops import einsum

from cs336_basics import function_utils
from cs336_basics.linear_module import Linear


class SwiGlu(nn.Module):
    def __init__(self,
    d_model: int,
    d_ff: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model,d_ff)
        self.w2 = Linear(d_ff,d_model)
        self.w3 = Linear(d_model,d_ff)

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.w2(functional_utils.silu(self.w1(x)) * self.w3(x))