import math

from jaxtyping import Float, Bool, Int
from typing import Optional
from torch import Tensor, tensor
from einops import rearrange, einsum
from collections.abc import Iterable
import torch



def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    tensor_out = in_features * torch.sigmoid(in_features)
    return tensor_out

def softmax(in_feature: torch.Tensor, dimension: int) -> torch.Tensor:
    max_value, _ = torch.max(in_feature, dim=dimension)
    # reshape shifted for broadcasting purpose
    shifted = in_feature - rearrange(max_value, "... -> ... 1")

    # reshape exp_sum: (...) -> (... 1) ; same as unsqueeze(-1)
    exp_sum = rearrange(torch.sum(torch.exp(shifted), dim=dimension), "... -> ... 1")
    prob = torch.exp(shifted) / exp_sum
    return prob

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values(=keys) d_v"],
    mask: Optional[Bool[Tensor, " ... queries keys"]],
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(tensor(Q.size(-1)))
    scores[~mask] = float("-inf")
    final_scores = softmax(scores, -1)
    attention = einsum(final_scores, V, "... queries key_values, ... key_values d_v -> ... queries d_v")
    return attention

def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    # logits: (batch seq vocab_size)
    # targets: (batch seq)
    negative_log_softmax_logits = - log_softmax(logits,-1)
    ce = torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))
    return ce

def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = 0.0
    for grad in grads:
        norm += (grad**2).sum()

    norm = math.sqrt(norm)

    clip_coef = min(1.0, max_l2_norm / (norm + eps))
    for grad in grads:
        grad.mul_(clip_coef)


