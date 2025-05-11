# from torch.utils.data import Dataset

# class TokenIdDataset(Dataset):

#     def __init__(self, data, block_size):
#         self.data = data
#         self.block_size = block_size

#     def __len__(self):
#         return len(self.data) - self.block_size

#     def __getitem__(self, index):
#         assert index < len(self.data) - self.block_size
#         x = self.data[index:index + self.block_size]
#         y = self.data[index + 1: index + 1 + self.block_size]
#         return x, y
# token_id_dataset.py

from torch.utils.data import Dataset
import torch
import random

class TokenIdDataset(Dataset):
    def __init__(self, token_sequences, block_size):
        """
        token_sequences: List[List[int]]
        """
        self.token_sequences = [seq for seq in token_sequences if len(seq) > block_size]
        self.block_size = block_size

    def __len__(self):
        # you can choose any sampling strategy here
        return sum(len(seq) - self.block_size for seq in self.token_sequences)

    def __getitem__(self, index):
        # randomly sample a valid article and offset
        while True:
            seq = random.choice(self.token_sequences)
            if len(seq) > self.block_size:
                break

        start = random.randint(0, len(seq) - self.block_size - 1)
        x = torch.tensor(seq[start:start + self.block_size], dtype=torch.long)
        y = torch.tensor(seq[start + 1:start + 1 + self.block_size], dtype=torch.long)
        return x, y
