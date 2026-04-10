import os
import re
import json
import math
import mlflow
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import islice
from torch.utils.data import IterableDataset, Dataset
from datasets import load_from_disk
from transformers import (
	AutoTokenizer,
	EarlyStoppingCallback,
	Trainer,
	TrainerCallback,
	TrainingArguments,
)
from safetensors.torch import load_file
from model.architecture import MidnightGPT

MLFLOW_EXPERIMENT_NAME = "midnight_huggingface_training"
MLFLOW_TRACKING_URI = "file:./mlruns"
RUN_NAME = "Midnight-GPT"
HTML_RE = re.compile(r"<(?!\|endoftext\|)[^>]+>")
CODE_RE = re.compile(r"(var\s+\w+\s*=|function\s*\(|document\.|console\.|=>|\{|\})")

os.makedirs("./mlruns", exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

CHECKPOINT_DIR = "./checkpoints"
OPTUNA_BEST_TRIAL_PATH = Path("./checkpoints/optuna/best_params_1775406019891.jsonl")
GEN_PROMPT = "Durante o período medieval"
CONTEXT_WINDOW = 512

CHECKPOINT="./checkpoints/training/second-iteration/checkpoint-74000/model.safetensors"


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

class StreamingTokenDataset(IterableDataset):
	def __init__(self, dataset, tokenizer, context_size):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.context_size = context_size
		self.total = 0
		self.total_noisy_data = 0
		self.total_clean_data = 0

	def __iter__(self):
		buffer = []
		
		for sample in self.dataset:
			self.total += 1
			text = "".join([
				"\n".join(p) + self.tokenizer.eos_token
				for p in sample["text"]["paragraphs"]
				if any(len(t.split()) > 1 for t in p)
			])
			if self.is_noisy_data(text):
				self.total_noisy_data += 1
				continue
			self.total_clean_data += 1
			if self.total >= 500 and self.total % 500 == 0:
				print("\nTotal records: ", self.total)
				print("Total noisy records: ",self.total_noisy_data)
				print("Total clean records: ",self.total_clean_data)

			tokens = self.tokenizer(text, use_fast=True, add_special_tokens=False)["input_ids"]

			buffer.extend(tokens)

			if len(buffer) > self.context_size * 4:
				random.shuffle(buffer)

			while len(buffer) >= self.context_size:
				chunk = buffer[:self.context_size]
				buffer = buffer[self.context_size:]

				yield {
					"input_ids": chunk
				}
	
	def is_noisy_data(self, text):
		if not text:
			return True

		if "<" in text and ">" in text:
			if HTML_RE.search(text):
				return True

		if CODE_RE.search(text):
			return True

		weird_ratio = sum(
			1 for c in text if not c.isalnum() and not c.isspace()
		) / len(text)

		if weird_ratio > 0.3:
			return True

		return False

class SimpleDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

class TokenCounterCallback(TrainerCallback):
	def __init__(self, context_size, batch_size, grad_accum):
		self.tokens_seen = 0
		self.tokens_per_step = context_size * batch_size * grad_accum

	def on_step_end(self, args, state, control, **kwargs):
		self.tokens_seen += self.tokens_per_step

		if state.global_step % 20 == 0:
			print(f"\nTokens seen: {self.tokens_seen:,}")

			if "mlflow" in args.report_to:
				mlflow.log_metric("tokens_seen", self.tokens_seen, step=state.global_step)

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

	raw_train_dataset = load_from_disk("./dataset/untokenized/brwac_train").shuffle(seed=42)
	train_dataset = StreamingTokenDataset(
		raw_train_dataset,
		tokenizer,
		CONTEXT_WINDOW
	)

	raw_val_dataset = load_from_disk("./dataset/untokenized/brwac_val").shuffle(seed=42)
	val_dataset = StreamingTokenDataset(
		raw_val_dataset,
		tokenizer,
		CONTEXT_WINDOW
	)

	raw_test_dataset = load_from_disk("./dataset/untokenized/brwac_test").shuffle(seed=42)
	test_dataset = StreamingTokenDataset(
		raw_test_dataset,
		tokenizer,
		CONTEXT_WINDOW
	)

	best_params = load_best_params(OPTUNA_BEST_TRIAL_PATH)
	print("Optuna best trial configuration:", best_params)

	val_eval_dataset = SimpleDataset(list(islice(val_dataset, 2000)))
	test_eval_dataset = SimpleDataset(list(islice(test_dataset, 2000)))

	model = MidnightGPT(config)
	state_dict = load_file(CHECKPOINT)
	model.load_state_dict(state_dict)

	use_fp16 = torch.cuda.is_available()
	dataloader_workers = min(2, os.cpu_count() or 1)

	training_args = TrainingArguments(
		output_dir="./checkpoints/training/third-iteration",
		max_steps = 10_000_000,
		num_train_epochs = 1,
		per_device_train_batch_size=best_params["batch_size"],
		per_device_eval_batch_size=best_params["batch_size"],
		gradient_accumulation_steps=best_params["grad_accum"],
		max_grad_norm=1.0,
		learning_rate=3.86e-5 * 0.2,
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
				CONTEXT_WINDOW,
				best_params["batch_size"],
				best_params["grad_accum"]
			)
		],
	)

	trainer.train()

	print("Calculating test loss")
	test_metrics = trainer.evaluate(test_eval_dataset)
	print("Final test loss:", test_metrics.get("eval_loss"))

	torch.save(trainer.model.state_dict(), "./model/dev/midnight_3.pt")


if __name__ == "__main__":
	main()
