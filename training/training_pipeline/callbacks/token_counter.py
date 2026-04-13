import torch
from transformers import TrainerCallback

class TokenCounterCallback(TrainerCallback):
	def __init__(self, context_size, batch_size, grad_accum, mlflow):
		self.tokens_seen = 0
		self.tokens_per_step = context_size * batch_size * grad_accum
		self.mlflow = mlflow

	def on_step_end(self, args, state, control, **kwargs):
		self.tokens_seen += self.tokens_per_step

		if state.global_step % 20 == 0:
			print(f"\nTokens seen: {self.tokens_seen:,}")

			if "mlflow" in args.report_to:
				self.mlflow.log_metric("tokens_seen", self.tokens_seen, step=state.global_step)