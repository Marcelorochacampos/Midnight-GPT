# python chunk.py --tokenizer=./checkpoints/tokenizer/tokenizer_20250508_145309.json --huggingface=wikipedia --subset=20220301.en
import os
import torch
import pandas as pd
import huggingface_hub
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datasets import load_dataset
from char_tokenizer import CharTokenizer
from token_id_dataset import TokenIdDataset
from demo_gpt import DemoGPT
from helper import get_cli_args
from train import train

huggingface_hub.HfApi().timeout = 60

EOS_TOKEN = "<|endoftext|>"

os.environ["HF_DATASETS_CACHE"] = "D:/huggingface"
os.environ["HF_HOME"] = "D:/huggingface"

def main():
    print("Starting the process to train the model.\n")
    args = get_cli_args()

    print("Arguments used for training\nDataset: {}\nTokenizer: {}\nModel: {}\nHuggingFace: {}\nOffset: {}\n".format(
        args.dataset, args.tokenizer, args.model, args.huggingface, getattr(args, 'offset', None)))
    print("")

    if not args.dataset and not args.huggingface:
        print("A dataset wasn't provided, stopping the process.\n")
        return

    is_cuda_available = torch.cuda.is_available()
    device = "cuda"
    print("Is cuda available: {}\n".format(is_cuda_available))

    if not is_cuda_available:
        print("Cuda is not available, stopping the process.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = CharTokenizer()

    if args.tokenizer:
        print("Loading stored state for tokenizer: {}\n".format(args.tokenizer))
        tokenizer.load_state(args.tokenizer)
    else:
        print("Tokenizer path not provided. Will train a new one if needed.\n")

    samples = []
    if args.dataset:
        text = Path(args.dataset).read_text(encoding='utf-8')
        samples = [line.strip() + EOS_TOKEN for line in text.split("\n\n") if line.strip()]
    elif args.huggingface:
        if not args.subset:
            raise ValueError("You must provide a --subset for the selected Hugging Face dataset")

        offset = getattr(args, 'offset', 0)
        chunk_size = 100
        range_start = offset * chunk_size
        range_end = (offset + 1) * chunk_size
        print("Dataset -\nRange Start: {}\nRange  End: {}".format(range_start, range_end))
        dataset = load_dataset(args.huggingface, args.subset, split=f"train[{range_start}:{range_end}]", cache_dir="D:/huggingface") 
        dataset = dataset.select(range(range_start, min(range_end, len(dataset))))

        for sample in dataset:
            text = sample["text"].strip()
            if text:
                samples.append(text + EOS_TOKEN)

    if not args.tokenizer:
        print("Training tokenizer from scratch.\n")
        tokenizer = CharTokenizer.train_from_text("\n".join(samples))
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
        "batch_size": 64,
        "train_iterations": 5000,
        "evaluation_interval": 100,
        "learning_rate": 4e-4,
        "epochs": 1,
        "num_workers": 8,
        "prefetch_factor": 4
    }

    model = DemoGPT(config).to(device)

    if args.model:
        print("Loading model state: {}\n".format(args.model))
        model.load_state_dict(torch.load(args.model))
    else:
        print("Model wasn't provided, creating from scratch.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"])

    # token_sequences = []
    # for text in samples:
    #     token_ids = tokenizer.encode(text)
    #     if len(token_ids) > 1:
    #         token_sequences.append(token_ids)

    # print(f"Tokenized {len(token_sequences)} samples for training.\n")

    # train_dataset = TokenIdDataset(token_sequences, block_size=config["context_size"])
    train_dataset = TokenIdDataset(samples, tokenizer, block_size=config["context_size"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=training_config["num_workers"],
        prefetch_factor=training_config["prefetch_factor"]
    )

    train(model, optimizer, tokenizer, train_dataloader, config, training_config)

if __name__ == "__main__":
    main()