import torch
from torch import nn

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.Q_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.K_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])
        self.V_weights = nn.Linear(config["embedding_dim"], config["head_size"], config["use_bias"])

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self, input):

        batch_size, tokens_num, embedding_dim = input.shape

        Q = self.Q_weights(input)
        K = self.K_weights(input)
        V = self.V_weights(input)

        attention_scores = Q @ K.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num,:tokens_num] == 0,
            -torch.inf
        )
        attention_scores = attention_scores / ( K.shape[-1] ** 0.5 )
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V