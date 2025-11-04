import einx
import torch
from einops import einsum
from torch import nn


class Rope(nn.Module):
    def __init__(self, theta: float, max_seq_len: int, dim: int, device=None):
        super().__init__()

        # position: (max_seq_len 1)
        position = torch.arange(max_seq_len).float().unsqueeze(1)

        # k: (1 d_k)
        k = torch.arange(1, dim // 2 + 1).float().unsqueeze(0)
        # calculate all angles: (max_seq_len d_k//2)
        angles = position / theta ** ((2 * k - 2) / dim)

        # pre-calculate sin and cos
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)

        # allocate sin_vals and cos_vals into rotation matrices
        rot_mats = torch.zeros(max_seq_len, dim, dim)

        rot_mats[:, torch.arange(0, dim, 2), torch.arange(0, dim, 2)] = cos_vals
        rot_mats[:, torch.arange(0, dim, 2), torch.arange(1, dim, 2)] = -sin_vals
        rot_mats[:, torch.arange(1, dim, 2), torch.arange(0, dim, 2)] = sin_vals
        rot_mats[:, torch.arange(1, dim, 2), torch.arange(1, dim, 2)] = cos_vals

        self.register_buffer("rot_mats", rot_mats, persistent=False)
        self.token_position : torch.Tensor
        self.theta = theta
        self.d_k = dim
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x: (batch seq_len d_k) ; token_position: (batch seq_len)

        # rot: (batch seq_len d_k d_k)
        rots = self.rot_mats[token_position]

        # apply rotation
        rot_x = einsum(rots, x, "... row col, ... col-> ... row")

        return rot_x

