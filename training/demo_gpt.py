import torch
from torch import nn

from transformer_block import TransformerBlock

class DemoGPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["embedding_dim"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["embedding_dim"])

        blocks = [TransformerBlock(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)

        self.layer_norm = nn.LayerNorm(config["embedding_dim"])
        self.unembedding = nn.Linear(config["embedding_dim"], config["vocabulary_size"], bias=False)

    def forward(self, input):
        batch_size, tokens_num = input.shape

        x = self.token_embedding_layer(input)
        sequence = torch.arange(tokens_num, device="cpu")
        x = x + self.positional_embedding_layer(sequence)

        x = self.layers(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)

        return x