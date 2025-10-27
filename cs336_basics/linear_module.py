import math
import torch
from torch import nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features,  device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))

        std = math.sqrt(2.0 / in_features + out_features)
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
        return Y