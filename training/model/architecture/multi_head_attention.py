import torch
from torch import nn

from .attention_head import AttentionHead

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        self.linear = nn.Linear(config["embedding_dim"], config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])


    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]

        scores_change = torch.cat(heads_outputs, dim=-1)

        scores_change = self.linear(scores_change)
        return self.dropout(scores_change)