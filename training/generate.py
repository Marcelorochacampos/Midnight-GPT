import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path

from char_tokenizer import CharTokenizer

from demo_gpt import DemoGPT
from helper import generate_with_prompt, get_cli_args

def main():
    print("Starting the process to train the model.\n")

    args = get_cli_args()

    print("Arguments used for training\nDataset: {}\nTokenizer: {}\nModel: {}\n".format(args.dataset, args.tokenizer, args.model))
    print("")

    if not args.model:
        print("A model path wasnt provided, stopping process.")
        return
    
    if not args.tokenizer:
        print("A tokenizer path wasnt provided, stopping process.")
        return

    is_cuda_available = torch.cuda.is_available()
    device = "cuda"
    print("Is cuda available: {}\n".format(is_cuda_available))

    if is_cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()        
    else:
        print("Cuda is not available, stopping the process.")
        return

    print("Loading stored state for tokenizer: {}\n".format(args.tokenizer))
    tokenizer = CharTokenizer()
    tokenizer.load_state(args.tokenizer)

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

    print("Loading model state: {}\n".format(args.model))
    model = DemoGPT(config).to(device)
    model.load_state_dict(torch.load(args.model))

    print(generate_with_prompt(model, config, tokenizer, "I have to go to work tomorrow, I forgot to pack food for lunch", max_tokens=5000))

if __name__ == "__main__":
    main()