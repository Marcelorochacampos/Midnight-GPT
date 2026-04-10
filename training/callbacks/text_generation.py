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