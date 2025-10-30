import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        std = 1
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3, b=3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        tensor_out = self.weights[token_ids]
        return tensor_out