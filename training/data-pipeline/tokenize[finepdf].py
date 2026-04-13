import os
import sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.configuration import load_global_configuration

# ================= CONFIG =================
config = load_global_configuration("./config/global_configuration.yaml")
model_config = config["model_architecture"]
CONTEXT_WINDOW = model_config["context_size"]

PREFIX = "finepdf"

train_path = f"./dataset/untokenized/{PREFIX}_train"
test_path = f"./dataset/untokenized/{PREFIX}_test"
val_path = f"./dataset/untokenized/{PREFIX}_val"

output_base = "./dataset/tokenized_fast"
os.makedirs(output_base, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

NUM_PROC = max(1, os.cpu_count() // 2)


# ================= TOKENIZE =================
def tokenize_function(examples):
    texts = [
        sentence + tokenizer.eos_token
        for text in examples["text"]
        for sentence in text.split("\n")
        if len(sentence.split()) > 1
    ]

    tokens = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False
    )

    tokens.pop("attention_mask", None)

    return tokens


# ================= GROUP INTO CHUNKS =================
def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    
    total_length = (len(concatenated) // CONTEXT_WINDOW) * CONTEXT_WINDOW
    
    result = {
        "input_ids": [
            concatenated[i:i + CONTEXT_WINDOW]
            for i in range(0, total_length, CONTEXT_WINDOW)
        ]
    }
    
    result["labels"] = result["input_ids"].copy()
    
    return result


# ================= PROCESS =================
def process(dataset_path, output_path, ds_type):
    print(f"\nProcessing {ds_type}")

    dataset = load_from_disk(dataset_path).shuffle(seed=42)
    dataset = dataset.select(range(400_000))

    print("Tokenizing...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    print("Grouping into chunks...")
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        desc="Grouping"
    )

    lm_dataset.save_to_disk(output_path)

    print(f"{ds_type} DONE → {output_path}")


# ================= MAIN =================
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