import torch as t
from torch.utils.data import Dataset
from src_reshaped.utils.vocab import Vocab
import pandas as pd


class TextSet(Dataset):
    def __init__(self, manifest, vocab_path='testing_vocab.model'):
        super(TextSet, self).__init__()
        self.vocab = Vocab(vocab_path)
        self.load_manifest(manifest)

    def load_manifest(self, manifest):
        self.df = pd.read_csv(manifest)
        self.df = self.df.reset_index()
        self.df = self.df[['wav_file', 'target']]
        self.df = self.df.to_dict('index')
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        target = self.df[item]['target']
        target = self.vocab.str2id(target)
        target_length = len(target)
        return target, target_length

if __name__ == '__main__':
    audioset = TextSet('data/manifest/ce_200.csv')
    next(iter(audioset))