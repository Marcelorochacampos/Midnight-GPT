from torch.utils.data import IterableDataset

class StreamingTokenizedDataset(IterableDataset):
	def __init__(self, dataset, tokenizer, context_size):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.context_size = context_size
		self.total = 0
		self.total_noisy_data = 0
		self.total_clean_data = 0

	def __iter__(self):
		for sample in self.dataset:
			self.total += 1
			if self.total % 1000 == 0:
				print("\nTotal records: ", self.total)
			yield {
				"input_ids": sample["input_ids"]
			}