import torch

class CausalLMDataCollator:
	def __call__(self, features):
		input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
		return {"input_ids": input_ids, "labels": input_ids.clone()}