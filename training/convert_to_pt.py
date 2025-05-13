import torch

from demo_gpt import DemoGPT
from char_tokenizer import CharTokenizer

def main():
    print("Converting model to .pt.\n")
    model_path = "./checkpoints/model/model_mad_20250511_235851_6000.pth"
    tokenizer = CharTokenizer()
    tokenizer.load_state("./checkpoints/tokenizer/tokenizer_20250508_145309.json")

    config = {
        "vocabulary_size": tokenizer.vocabulary_size(),
        "context_size": 256,
        "embedding_dim": 768,
        "heads_num": 12,
        "layers_num": 10,
        "dropout_rate": 0.1,
        "use_bias": False
    }

    config["head_size"] = config["embedding_dim"] // config["heads_num"]

    print("Loading model state: {}\n".format(model_path))
    model = DemoGPT(config).to("cpu")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randint(0, config["vocabulary_size"], (1, config["context_size"]))
    dummy_input = dummy_input.to("cpu")
    # Convert to TorchScript
    print("Tracing model and saving to .pt...\n")
    traced = torch.jit.trace(model, dummy_input)
    traced.save("midnight_gpt_6000_01_cpu.pt")
    print("Model saved as midnight_gpt_6000_01_cpu.pt\n")
    return

if __name__ == "__main__":
    main()