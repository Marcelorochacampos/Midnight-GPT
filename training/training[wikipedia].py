import json
import math
import os
import mlflow
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
	AutoTokenizer,
	EarlyStoppingCallback,
	Trainer,
	TrainerCallback,
	TrainingArguments,
)

from model.architecture import MidnightGPT

MLFLOW_EXPERIMENT_NAME = "midnight_huggingface_training"
MLFLOW_TRACKING_URI = "file:./mlruns"
RUN_NAME = "Midnight-GPT"

os.makedirs("./mlruns", exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

CHECKPOINT_DIR = "./checkpoints"
OPTUNA_BEST_TRIAL_PATH = Path("./checkpoints/optuna/best_params_1775406019891.jsonl")
GEN_PROMPT = "Durante o período medieval "
CONTEXT_WINDOW = 512
TOTAL_TRAIN_TOKENS = 300_000_000


class CausalLMDataCollator:
	def __call__(self, features):
		input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
		return {"input_ids": input_ids, "labels": input_ids.clone()}


class MidnightTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
		input_ids = inputs["input_ids"]
		output = model(input_ids=input_ids, labels=input_ids)
		if return_outputs:
			return output.loss, output
		return output.loss


class TextGenerationCallback(TrainerCallback):
	def __init__(self, tokenizer, config, prompt, max_new_tokens=50, temperature=0.8):
		self.tokenizer = tokenizer
		self.config = config
		self.prompt = prompt
		self.max_new_tokens = max_new_tokens
		self.temperature = temperature

	@torch.no_grad()
	def _generate(self, model):
		model.eval()
		model_device = next(model.parameters()).device
		input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(model_device)

		for _ in range(self.max_new_tokens):
			input_ids = input_ids[:, -self.config["context_size"] :]
			logits = model(input_ids=input_ids, labels=input_ids).logits
			probs = torch.softmax(logits[:, -1, :] / self.temperature, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			input_ids = torch.cat([input_ids, next_token], dim=1)

		return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

	def on_evaluate(self, args, state, control, model=None, **kwargs):
		if model is None:
			return
		generated = self._generate(model)
		print("-----")
		print(generated)
		print("-----")


def load_best_params(path):
	records = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			records.append(json.loads(line))

	if not records:
		raise ValueError(f"No Optuna trial records found in {path}")

	return records[0]["params"]


def main():
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)

	tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
	tokenizer.pad_token = tokenizer.eos_token

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

	params_count = (
		config["vocabulary_size"] * config["embedding_dim"]
		+ config["context_size"] * config["embedding_dim"]
		+ config["layers_num"] * (12 * config["embedding_dim"] * config["embedding_dim"])
	)

	print("==============================")
	print("Model configuration:", config)
	print(f"Model size (parameters): {params_count:,}")
	print("==============================")

	train_dataset = load_from_disk("./dataset/tokenized/train_512")
	val_dataset = load_from_disk("./dataset/tokenized/val_512")
	test_dataset = load_from_disk("./dataset/tokenized/test_512")

	best_params = load_best_params(OPTUNA_BEST_TRIAL_PATH)
	print("Optuna best trial configuration:", best_params)

	max_train_steps = math.ceil(
		TOTAL_TRAIN_TOKENS / (best_params["batch_size"] * CONTEXT_WINDOW)
	)
	print("Max train steps:", max_train_steps)
	warmup_steps = int(0.03 * max_train_steps)

	eval_subset_size = min(len(val_dataset), best_params["batch_size"] * 21)
	val_eval_dataset = val_dataset.select(range(eval_subset_size))

	model = MidnightGPT(config)

	use_fp16 = torch.cuda.is_available()
	dataloader_workers = min(2, os.cpu_count() or 1)

	training_args = TrainingArguments(
		output_dir="./checkpoints/training",
		max_steps=max_train_steps,
		per_device_train_batch_size=best_params["batch_size"],
		per_device_eval_batch_size=best_params["batch_size"],
		gradient_accumulation_steps=best_params["grad_accum"],
		max_grad_norm=1.0,
		learning_rate=best_params["lr"],
		lr_scheduler_type="cosine",
		warmup_steps=warmup_steps,
		weight_decay=best_params["weight_decay"],
		eval_strategy="steps",
		eval_steps=500,
		save_strategy="steps",
		save_steps=500,
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
			EarlyStoppingCallback(early_stopping_patience=10),
		],
	)

	trainer.train()

	print("Calculating test loss")
	test_metrics = trainer.evaluate(test_dataset)
	print("Final test loss:", test_metrics.get("eval_loss"))

	torch.save(trainer.model.state_dict(), "./model/dev/midnight_gpt.pt")


if __name__ == "__main__":
	main()
