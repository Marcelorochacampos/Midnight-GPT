import torch
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from huggingface_hub import login
from model.architecture import MidnightGPT
from utils.configuration import load_global_configuration

class MidnightConfig(PretrainedConfig):
    model_type = "midnight_gpt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.midnight_config = kwargs


class MidnightHFModel(PreTrainedModel):
    config_class = MidnightConfig

    def __init__(self, config):
        super().__init__(config)

        # modelo original (intacto)
        self.model = MidnightGPT(config.midnight_config)

        self.post_init()

    def forward(self, **kwargs):
        return self.model(**kwargs)


MODEL_PATH = "./model/dev/midnight_4.pt"
REPO_ID = "Heyves/MidnightGPT-case-study"

GLOBAL_CONFIGURATION = load_global_configuration("./config/global_configuration.yaml")
config = GLOBAL_CONFIGURATION["model_architecture"]

login()

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

hf_config = MidnightConfig(**config)
model = MidnightHFModel(hf_config)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.model.load_state_dict(state_dict)

model.eval()

SAVE_DIR = "./model/prod/MidnightGPT-case-study"

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)

print("✅ Modelo enviado com sucesso!")