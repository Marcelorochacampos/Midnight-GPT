import os
import json
import math
import mlflow
import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import islice
from datasets import load_from_disk, interleave_datasets
from transformers import (
	AutoTokenizer,
	EarlyStoppingCallback,
	TrainingArguments,
)
from safetensors.torch import load_file
from utils.configuration import load_global_configuration
from model.architecture import MidnightGPT
from training_pipeline.dataset import StreamingTokenizedDataset, SimpleDataset
from training_pipeline.trainer import MidnightTrainer
from training_pipeline.callbacks import SaveBestModelCallback, TextGenerationCallback, TokenCounterCallback
from training_pipeline.collate import CausalLMDataCollator


MLFLOW_EXPERIMENT_NAME = "midnight_huggingface_training"
MLFLOW_TRACKING_URI = "file:./mlruns"
RUN_NAME = "Midnight-GPT-MIX"

os.makedirs("./mlruns", exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

OPTUNA_BEST_TRIAL_PATH = Path("./checkpoints/optuna/best_params_1775406019891.jsonl")
GEN_PROMPT = "Durante o período medieval"

CHECKPOINT="./checkpoints/training/third-iteration/checkpoint-74000/model.safetensors"
GLOBAL_CONFIGURATION = load_global_configuration("./config/global_configuration.yaml")


def load_best_params(path):
	records = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			records.append(json.loads(line))

	if not records:
		raise ValueError(f"No Optuna trial records found in {path}")

	return records[0]["params"]

def main():
	tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
	tokenizer.pad_token = tokenizer.eos_token

	config = GLOBAL_CONFIGURATION["model_architecture"]

	params_count = (
		config["vocabulary_size"] * config["embedding_dim"]
		+ config["context_size"] * config["embedding_dim"]
		+ config["layers_num"] * (12 * config["embedding_dim"] * config["embedding_dim"])
	)

	print("==============================")
	print("Model configuration:", config)
	print(f"Model size (parameters): {params_count:,}")
	print("==============================")

	raw_finepdf_train_dataset = load_from_disk("./dataset/tokenized/finepdf-train-512").shuffle(seed=42)
	raw_wikipedia_train_dataset = load_from_disk("./dataset/tokenized/wikipedia-train-512").shuffle(seed=42)

	raw_train_dataset = interleave_datasets(
		[
			raw_wikipedia_train_dataset,
			raw_finepdf_train_dataset
		],
		probabilities=[0.6, 0.4],
		seed=42,
		stopping_strategy="first_exhausted"
	)
	
	train_dataset = StreamingTokenizedDataset(
		raw_train_dataset,
		tokenizer,
		config["context_size"]
	)

	raw_finepdf_val_dataset = load_from_disk("./dataset/tokenized/finepdf-val-512").shuffle(seed=42)
	raw_wikipedia_val_dataset = load_from_disk("./dataset/tokenized/wikipedia-val-512").shuffle(seed=42)

	raw_val_dataset = interleave_datasets(
		[
			raw_wikipedia_val_dataset,
			raw_finepdf_val_dataset
		],
		probabilities=[0.6, 0.4],
		seed=42,
		stopping_strategy="first_exhausted"
	)

	val_dataset = StreamingTokenizedDataset(
		raw_val_dataset,
		tokenizer,
		config["context_size"]
	)

	raw_finepdf_test_dataset = load_from_disk("./dataset/tokenized/finepdf-test-512").shuffle(seed=42)
	raw_wikipedia_test_dataset = load_from_disk("./dataset/tokenized/wikipedia-test-512").shuffle(seed=42)

	raw_test_dataset = interleave_datasets(
		[
			raw_wikipedia_test_dataset,
			raw_finepdf_test_dataset
		],
		probabilities=[0.6, 0.4],
		seed=42,
		stopping_strategy="first_exhausted"
	)

	test_dataset = StreamingTokenizedDataset(
		raw_test_dataset,
		tokenizer,
		config["context_size"]
	)

	best_params = load_best_params(OPTUNA_BEST_TRIAL_PATH)
	print("Optuna best trial configuration:", best_params)

	val_eval_dataset = SimpleDataset(list(islice(val_dataset, 10000)))
	test_eval_dataset = SimpleDataset(list(islice(test_dataset, 10000)))

	model = MidnightGPT(config)
	state_dict = load_file(CHECKPOINT)
	model.load_state_dict(state_dict)

	use_fp16 = torch.cuda.is_available()
	dataloader_workers = min(2, os.cpu_count() or 1)

	training_args = TrainingArguments(
		output_dir="./checkpoints/training/fourth-iteration",
		max_steps = 10_000_000,
		num_train_epochs = 1,
		per_device_train_batch_size=best_params["batch_size"],
		per_device_eval_batch_size=best_params["batch_size"],
		gradient_accumulation_steps=best_params["grad_accum"],
		max_grad_norm=1.0,
		learning_rate=7e-6,
		lr_scheduler_type="cosine",
		warmup_steps = 300,
		weight_decay=best_params["weight_decay"],
		eval_strategy="steps",
		eval_steps=2000,
		save_strategy="steps",
		save_steps=2000,
		save_total_limit=2,
		logging_strategy="steps",
		logging_steps=20,
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		dataloader_num_workers=dataloader_workers,
		dataloader_pin_memory=torch.cuda.is_available(),
		fp16=use_fp16,
		report_to=["mlflow"],
		run_name=RUN_NAME,
		remove_unused_columns=False,
	)

	trainer = MidnightTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_eval_dataset,
		data_collator=CausalLMDataCollator(),
		callbacks=[
			TextGenerationCallback(tokenizer, config, GEN_PROMPT),
			EarlyStoppingCallback(early_stopping_patience=20),
			TokenCounterCallback(
				config["context_size"],
				best_params["batch_size"],
				best_params["grad_accum"],
				mlflow
			),
			SaveBestModelCallback("./model/dev/midnight_4_backup.pt")
		],
	)

	trainer.train()

	print("Calculating test loss")
	test_metrics = trainer.evaluate(test_eval_dataset)
	print("Final test loss:", test_metrics.get("eval_loss"))

	torch.save(trainer.model.state_dict(), "./model/dev/midnight_4.pt")


if __name__ == "__main__":
	main()
