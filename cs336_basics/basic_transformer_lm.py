from torch import nn

from cs336_basics import rope
from cs336_basics.embedding_module import Embedding
from cs336_basics.linear_module import Linear
from cs336_basics.rmsnorm_module import RMSNorm
from cs336_basics.rope import Rope
from cs336_basics.transformer_block import TransformerBlock
from torch import Tensor
from jaxtyping import Float, Bool, Int
import logging

logger = logging.getLogger(__name__)

class BasicTransformerLM(nn.Module):
    """ A Transformer Language Model
    Args:
        vocab_size: int,
            - vocabulary size for the model
        context_length: int,
            - maximum context length
        d_model: int,
            - token / embedding dimension
        num_layers: int,
            - number of transformer blocks
        d_ff: int,
            - dimension for feed-forward layers
                (normally 4*d_model; 8/3*d_model for multi-head attention)
        num_heads: int,
            - number of attention heads
                ('d_model' must be evenly divisible by num_heads)
        rope_theta: int,
            - THETA for ROPE positional encoding

    Returns:
        FloatTensor of shape (batch_size, seq_len, vocab_size) with
        the predicted unnormalized next-word distribution for each token
    """

    def __init__(self, vocab_size, context_length, d_model, num_layers, d_ff, num_heads, rope_theta):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }

        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = Rope(
            theta=rope_theta,
            max_seq_len=context_length,
            dim=d_head
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count(default), with lm_head being subtracted.
        """
        n_para = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_para -= self.lm_head.weight.numel()

        return n_para

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        """
        Args:
            x: Input token IDs for the language model

        Returns:
            FloatTensor
            of shape (batch_size, seq_len, vocab_size) with
            the predicted unnormalized next-word distribution
            for each token
        """

        _, seq_len = x.shape
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x

