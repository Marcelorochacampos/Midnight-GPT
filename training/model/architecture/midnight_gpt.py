import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from .transformer_block import TransformerBlock

class MidnightGPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["embedding_dim"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["embedding_dim"])

        blocks = [TransformerBlock(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)

        self.layer_norm = nn.LayerNorm(config["embedding_dim"])
        self.output = nn.Linear(config["embedding_dim"], config["vocabulary_size"], bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        batch_size, tokens_num = input_ids.shape
        device = input_ids.device

        x = self.token_embedding_layer(input_ids)

        positions = torch.arange(tokens_num, device=device)
        x = x + self.positional_embedding_layer(positions)

        x = self.layers(x)

        x = self.layer_norm(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )