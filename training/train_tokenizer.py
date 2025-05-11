import os

os.environ["HF_DATASETS_CACHE"] = "D:/huggingface"
os.environ["HF_HOME"] = "D:/huggingface"

import pandas as pd

from pathlib import Path
from datasets import load_dataset
from char_tokenizer import CharTokenizer
from helper import get_cli_args

EOS_TOKEN = "<|endoftext|>"

def main():
    print("Starting the process to train the tokenizer.\n")

    args = get_cli_args()

    print("Arguments used for training\nDataset: {}\nTokenizer: {}\nModel: {}\nHuggingFace: {}\n".format(args.dataset, args.tokenizer, args.model, args.huggingface))
    print("")

    if not args.dataset and not args.huggingface:
        print("A dataset wasnt provided, stopping the process.\n")
        return
    
    tokenizer = CharTokenizer()
    if args.tokenizer:
        print("Loading stored state for tokenizer: {}\n".format(args.tokenizer))
        tokenizer.load_state(args.tokenizer)
    
    if args.dataset:
        text = Path(args.dataset).read_text(encoding='utf-8')

        if args.tokenizer:
            tokenizer.update_training(text)
            tokenizer.save_state()
        else:
            print("Loading tokenizer from scratch\n")
            tokenizer = CharTokenizer.train_from_text(text)
            tokenizer.save_state()

    elif args.huggingface:
        if not args.subset:
            raise ValueError("You must provide a --subset for the selected Hugging Face dataset")
        dataset = load_dataset(args.huggingface, args.subset, split="train", trust_remote_code=True)
        # Set to store all unique characters
        vocab_set = set()

        for i, sample in enumerate(dataset):
            text = sample["text"].strip() + EOS_TOKEN
            vocab_set.update(text)

            if i % 100_000 == 0:
                print(f"Processed {i} articles")
                tokenizer = CharTokenizer(sorted(vocab_set))
                tokenizer.save_state()

        # Train tokenizer with full vocabulary after loop
        tokenizer = CharTokenizer(sorted(vocab_set))
        tokenizer.save_state()
        print("Tokenizer training complete. Vocabulary size:", tokenizer.vocabulary_size())

    
    
    


if __name__ == "__main__":
    main()