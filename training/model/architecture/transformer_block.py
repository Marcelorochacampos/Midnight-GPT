from torch import nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embedding_dim"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embedding_dim"])

    def forward(self, input):
        residual = input
        x = self.multi_head(self.layer_norm_1(input))
        x = x + residual

        residual = x
        x = self.feed_forward(self.layer_norm_2(input))
        return x + residual