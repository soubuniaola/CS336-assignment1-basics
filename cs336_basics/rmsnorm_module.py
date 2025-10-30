import torch
from torch import nn
from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch seq d_model)
        in_dtype = x.dtype
        x.to(torch.float32)
        rms = reduce(x**2, "batch seq d_model -> batch seq 1", 'mean')
        tensor_out = x * self.gamma / torch.sqrt(rms+self.eps)
        return tensor_out.to(in_dtype)
