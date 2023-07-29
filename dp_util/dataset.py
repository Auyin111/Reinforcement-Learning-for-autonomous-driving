import numpy as np
import torch


class CustomDataset(torch.utils.data.dataset.Dataset):
    """info: is a tuple and it element can be string data such as id or other identifier"""
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target, info = self.dataset[index]
        return np.array(example), target, info

    def __len__(self):
        return len(self.dataset)