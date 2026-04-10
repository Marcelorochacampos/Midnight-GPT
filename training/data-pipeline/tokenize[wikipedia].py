import os
import sys
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.configuration import load_global_configuration

# ================= CONFIG =================
config = load_global_configuration("./config/global_configuration.yaml")
model_config = config["model_architecture"]
CONTEXT_WINDOW = model_config["context_size"]

PREFIX = "wikipedia"

train_path = f"./dataset/untokenized/{PREFIX}_pt_train"
test_path = f"./dataset/untokenized/{PREFIX}_pt_test"
val_path = f"./dataset/untokenized/{PREFIX}_pt_val"

output_base = "./dataset/tokenized_arrow_streaming"
os.makedirs(output_base, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

def generate_examples(dataset):
    buffer = []

    for example in dataset:
        text = example["text"] + tokenizer.eos_token

        tokens = tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

        buffer.extend(tokens)

        while len(buffer) >= CONTEXT_WINDOW:
            chunk = buffer[:CONTEXT_WINDOW]
            buffer = buffer[CONTEXT_WINDOW:]

            yield {
                "input_ids": chunk,
                "labels": chunk
            }

def process(dataset_path, output_path, ds_type):
    print(f"\nProcessing {ds_type} → Arrow streaming")

    dataset = load_from_disk(dataset_path)

    arrow_ds = Dataset.from_generator(
        lambda: generate_examples(dataset)
    )

    arrow_ds.save_to_disk(output_path)

    print(f"{ds_type} DONE → {output_path}")

def main():
    process(
        train_path,
        f"{output_base}/{PREFIX}-train-{CONTEXT_WINDOW}",
        "TRAIN"
    )

    process(
        test_path,
        f"{output_base}/{PREFIX}-test-{CONTEXT_WINDOW}",
        "TEST"
    )

    process(
        val_path,
        f"{output_base}/{PREFIX}-val-{CONTEXT_WINDOW}",
        "VAL"
    )


if __name__ == "__main__":
    main()