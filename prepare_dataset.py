import math
import logging

import copy

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))

        data_size, vocab_size = len(data), len(chars)
        logger.info('data has %d characters, %d unique.'%(data_size, vocab_size))

        self.tokenizer = {ch:i for i, ch in enumerate(chars)}
        self.decoder = {i:ch for i,ch in enumerate(chars)}

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size]

        encoding = torch.tensor([self.tokenizer[c] for c in chunk], dtype=torch.long)
        
        input_ids = encoding[:-1]
        labels = encoding[1:]

        return input_ids, labels #x,y

if __name__ == "__main__":
    pth = "./tinyshakespeare.txt"
    with open(pth, 'r', encoding='utf-8') as f:
        text = f.read()
    
    block_size = 256
    dataset = CharDataset(text, block_size)
    print("len", len(dataset))
    print("input_ids", dataset[0][0].shape)
    print("labels", dataset[0][1].shape)
    print(" text sample\n", ''.join([dataset.decoder[c.item()] for c in dataset[0][0][:20]]))
    print(" label sample\n", ''.join([dataset.decoder[c.item()] for c in dataset[0][1][:20]]))