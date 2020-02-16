import torch as t
from torch.utils.data import DataLoader, Dataset


class LMDataSet(Dataset):
    def __init__(self):
        super(LMDataSet, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
