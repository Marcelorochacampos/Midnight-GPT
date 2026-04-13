from transformers import Trainer

class MidnightTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
		input_ids = inputs["input_ids"]
		output = model(input_ids=input_ids, labels=input_ids)
		if return_outputs:
			return output.loss, output
		return output.loss