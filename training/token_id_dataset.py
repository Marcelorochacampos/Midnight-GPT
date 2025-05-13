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

# from torch.utils.data import Dataset
# import torch
# import random

# class TokenIdDataset(Dataset):
#     def __init__(self, token_sequences, block_size):
#         """
#         token_sequences: List[List[int]]
#         """
#         self.token_sequences = [seq for seq in token_sequences if len(seq) > block_size]
#         self.block_size = block_size

#     def __len__(self):
#         # you can choose any sampling strategy here
#         return sum(len(seq) - self.block_size for seq in self.token_sequences)

#     def __getitem__(self, index):
#         # randomly sample a valid article and offset
#         while True:
#             seq = random.choice(self.token_sequences)
#             if len(seq) > self.block_size:
#                 break

#         start = random.randint(0, len(seq) - self.block_size - 1)
#         x = torch.tensor(seq[start:start + self.block_size], dtype=torch.long)
#         y = torch.tensor(seq[start + 1:start + 1 + self.block_size], dtype=torch.long)
#         return x, y

import torch
class TokenIdDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)

        # Truncate or pad to block_size
        if len(token_ids) > self.block_size:
            token_ids = token_ids[:self.block_size]
        elif len(token_ids) < self.block_size:
            token_ids += [0] * (self.block_size - len(token_ids))  # Padding

        # Convert to tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.clone().detach().to(torch.long)
        else:
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return token_ids

