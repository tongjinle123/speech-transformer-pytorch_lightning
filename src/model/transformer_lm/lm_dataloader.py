import torch as t
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from src.utils.vocab import Vocab


class LMDataSet(Dataset):
    def __init__(self, manifest_list, vocab_path, max_length):
        super(LMDataSet, self).__init__()
        self.df = load_manifests(manifest_list)
        self.df['target'] = self.df.target.apply(self.vocab.str2id)
        self.vocab = Vocab(vocab_path)

        self.length = len(self.df)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def load_manifests(manifest_list):
    df = pd.DataFrame()
    for i in manifest_list:
        ndf = pd.read_csv(i)
        df = pd.concat([df, ndf])
    return df
