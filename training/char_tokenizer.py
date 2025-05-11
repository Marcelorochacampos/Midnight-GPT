import os
import json
import torch
from datetime import datetime

class CharTokenizer():

    def __init__(self, vocabulary = None):

        self.token_id_for_char = {}
        self.char_for_token_id = {}

        if vocabulary:
            self.token_id_for_char = { char: token_id for token_id, char in enumerate(vocabulary) }
            self.char_for_token_id = { token_id: char for token_id, char in enumerate(vocabulary) }

    def load_state(self, path):
        with open(path, "r") as file:
            state = json.load(file)
            self.token_id_for_char = state["token_id_for_char"]
            self.char_for_token_id = {int(k): v for k, v in state["char_for_token_id"].items()}

    def save_state(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tokenizer_name = "tokenizer_{}.json".format(timestamp)
        path = "./checkpoints/tokenizer/{}".format(tokenizer_name)
        print("Storing tokenizer checkpoint: {}\n".format(path))

        os.makedirs("./checkpoints/tokenizer", exist_ok=True)

        state = {
            "token_id_for_char": self.token_id_for_char,
            "char_for_token_id": self.char_for_token_id
        }
        
        with open(path, "w") as f:
            json.dump(state, f)

    def update_training(self, text):
        new_chars = set(text) - set(self.token_id_for_char.keys())
        next_token_id = len(self.token_id_for_char)

        for char in sorted(new_chars):
            self.token_id_for_char[char] = next_token_id
            self.char_for_token_id[next_token_id] = char
            next_token_id += 1

    @staticmethod
    def train_from_text(text):
        vocabulary = set(text)
        return CharTokenizer(sorted(list(vocabulary)))

    def encode(self, text):
        token_ids = []
        for char in text:
            token_ids.append(self.token_id_for_char[char])
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        chars = []
        for token_id in token_ids.tolist():
            chars.append(self.char_for_token_id[token_id])
        return ''.join(chars)

    def vocabulary_size(self):
        return len(self.token_id_for_char)