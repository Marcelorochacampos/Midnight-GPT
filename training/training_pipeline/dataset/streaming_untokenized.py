import random
import re
from torch.utils.data import IterableDataset

HTML_RE = re.compile(r"<(?!\|endoftext\|)[^>]+>")
CODE_RE = re.compile(r"(var\s+\w+\s*=|function\s*\(|document\.|console\.|=>|\{|\})")
TABLE_RE = re.compile(r"\|.*\|")
MULTI_PIPE_RE = re.compile(r"\|{2,}")
REPEATED_TOKEN_RE = re.compile(r"\b(\w+)\b(?:.*\b\1\b){5,}", re.IGNORECASE)

class StreamingUntokenDataset(IterableDataset):
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
			text = sample["text"] + self.tokenizer.eos_token 

			if self.is_noisy_data(text):
				self.total_noisy_data += 1
				print("Noisy text: {}".format(text[:1000]))
				continue
			self.total_clean_data += 1
			if self.total % 500 == 0:
				print("\nTotal records: ", self.total)
				print("Total noisy records: ",self.total_noisy_data)
				print("Total clean records: ",self.total_clean_data)

			tokens = self.tokenizer(text, use_fast=True, add_special_tokens=False)["input_ids"]

			buffer.extend(tokens)

			while len(buffer) >= self.context_size:
				chunk = buffer[:self.context_size]
				buffer = buffer[self.context_size:]

				yield {
					"input_ids": chunk
				}
	
	def is_noisy_data(self, text):
		if not text or len(text) < 20:
			return True

		if "<" in text and ">" in text:
			if HTML_RE.search(text):
				return True

		if CODE_RE.search(text):
			return True

		pipe_count = text.count("|")
		if pipe_count > 5:
			return True

		if TABLE_RE.search(text):
			return True

		lines = text.split("\n")
		short_lines = sum(1 for l in lines if len(l.split()) <= 3)
		if len(lines) > 5 and short_lines / len(lines) > 0.6:
			return True

		if REPEATED_TOKEN_RE.search(text):
			return True

		weird_ratio = sum(
			1 for c in text if not c.isalnum() and not c.isspace()
		) / len(text)

		if weird_ratio > 0.3:
			return True

		words = text.split()
		if len(words) > 0:
			avg_word_len = sum(len(w) for w in words) / len(words)
			if avg_word_len < 3:  # muito fragmentado tipo "AI 1"
				return True

		return False