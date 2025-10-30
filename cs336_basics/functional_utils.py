from jaxtyping import Float, Bool
from torch import Tensor, tensor
from einops import rearrange, einsum
import torch



def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    tensor_out = in_features * torch.sigmoid(in_features)
    return tensor_out

def softmax(in_feature: torch.Tensor, dimension: int) -> torch.Tensor:
    max_value, indices = torch.max(in_feature, dim=dimension)
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
    mask: Bool[Tensor, " ... queries keys"] | None= None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(tensor(Q.size(-1)))
    scores[~mask] = float("-inf")
    final_scores = softmax(scores, -1)
    attention = einsum(final_scores, V, "... queries key_values, ... key_values d_v -> ... queries d_v")
    return attention