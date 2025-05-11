import os

os.environ["HF_DATASETS_CACHE"] = "D:/huggingface"
os.environ["HF_HOME"] = "D:/huggingface"

import torch
import pandas as pd
import huggingface_hub
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from datasets import load_dataset

from char_tokenizer import CharTokenizer
from token_id_dataset import TokenIdDataset

from demo_gpt import DemoGPT
from helper import get_cli_args
from train import train

huggingface_hub.HfApi().timeout = 60

EOS_TOKEN = "<|endoftext|>"

def main():
    print("Starting the process to train the model.\n")

    args = get_cli_args()

    print("Arguments used for training\nDataset: {}\nTokenizer: {}\nModel: {}\nHuggingFace: {}\n".format(args.dataset, args.tokenizer, args.model, args.huggingface))
    print("")

    if not args.dataset and not args.huggingface:
        print("A dataset wasnt provided, stopping the process.\n")
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

    if args.dataset:
        text = Path(args.dataset).read_text(encoding='utf-8')
    elif args.huggingface:
        if not args.subset:
            raise ValueError("You must provide a --subset for the selected Hugging Face dataset")
        # Support chunked loading by offset
        offset = getattr(args, 'offset', 0)
        chunk_size = 10
        range_start = offset * chunk_size
        range_end = (offset + 1) * chunk_size
        dataset = load_dataset(args.huggingface, args.subset, split=f"train[{range_start}:{range_end}]", trust_remote_code=True)

        df = pd.DataFrame(dataset)
        print(df.head())
        text = '\n'.join([sample["text"].strip() + EOS_TOKEN for sample in dataset])

    tokenizer = CharTokenizer()

    if args.tokenizer:
        print("Loading stored state for tokenizer: {}\n".format(args.tokenizer))
        tokenizer.load_state(args.tokenizer)
    else:
        print("Loading tokenizer from scratch\n")
        tokenizer = CharTokenizer.train_from_text(text)
        tokenizer.save_state()

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

    training_config = {
        "batch_size": 32,
        "train_iterations": 5000,
        "evaluation_interval": 100,
        "learning_rate": 4e-4
    }

    model = DemoGPT(config).to(device)

    if args.model:
        print("Loading model state: {}\n".format(args.model))
        model.load_state_dict(torch.load(args.model))
    else:
        print("Model wasnt provided, creating from scratch.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"])

    train_data = tokenizer.encode(text).to(device)
    train_dataset = TokenIdDataset(train_data, block_size=256)
    train_dataloader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)

    train(model, optimizer, tokenizer, train_dataloader, config, training_config)

if __name__ == "__main__":
    main()