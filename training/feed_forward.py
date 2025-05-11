from torch import nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"] * 4, config["embedding_dim"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)