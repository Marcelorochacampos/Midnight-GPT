import torch
from safetensors.torch import load_file
from model.architecture import MidnightGPT

CHECKPOINT="./checkpoints/training/third-iteration/checkpoint-74000/model.safetensors"

config = {
    "vocabulary_size": 50257,
    "context_size": 512,
    "embedding_dim": 768,
    "heads_num": 12,
    "layers_num": 10,
    "dropout_rate": 0.1,
    "use_bias": False
}
config["head_size"] = config["embedding_dim"] // config["heads_num"]

model = MidnightGPT(config)
state_dict = load_file(CHECKPOINT)
model.load_state_dict(state_dict)

torch.save(model.state_dict(), "./model/dev/midnight_3.pt")