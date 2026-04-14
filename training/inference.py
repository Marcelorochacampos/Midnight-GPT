import argparse

import torch
from transformers import AutoTokenizer

from model.architecture import MidnightGPT
from utils.configuration import load_global_configuration


def generate_text(model, tokenizer, prompt, context_size, temperature, max_new_tokens):
	model.eval()
	device = next(model.parameters()).device
	input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
	print("Max new tokens ", max_new_tokens)

	for _ in range(max_new_tokens):
		input_window = input_ids[:, -context_size:]
		with torch.no_grad():
			logits = model(input_ids=input_window).logits
			probs = torch.softmax(logits[:, -1, :], dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)

		input_ids = torch.cat([input_ids, next_token], dim=1)

	return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
	parser = argparse.ArgumentParser(description="Run inference with Midnight-GPT")
	parser.add_argument(
		"--prompt",
		type=str,
		default="O período medieval foi marcado pelo",
		help="Prompt to start generation",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=None,
		help="Override max_new_tokens from configuration",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1,
		help="Override temperature from configuration",
	)
	args = parser.parse_args()

	config = load_global_configuration("./config/global_configuration.yaml")
	model_config = config["model_architecture"]
	inference_config = config.get("inference", {})
	model_path = config["paths"]["model"]["dev_weights"]

	temperature = args.temperature if args.temperature is not None else inference_config.get("temperature", 1)
	max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else inference_config.get("max_new_tokens", 10000)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

	model = MidnightGPT(model_config).to(device)
	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict)

	output = generate_text(
		model=model,
		tokenizer=tokenizer,
		prompt=args.prompt,
		context_size=model_config["context_size"],
		temperature=temperature,
		max_new_tokens=max_new_tokens,
	)

	print("\n=== Prompt ===")
	print(args.prompt)
	print("\n=== Generated ===")
	print(output)


if __name__ == "__main__":
	main()
