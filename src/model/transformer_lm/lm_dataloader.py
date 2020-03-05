import torch as t
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from src.utils.vocab import Vocab


class LMDataSet(Dataset):
    def __init__(self, manifest_list, vocab_path):
        super(LMDataSet, self).__init__()
        self.df = load_manifests(manifest_list)
        self.vocab = Vocab(vocab_path)
        self.df = self.df[['target']]
        self.df['target_word'] = self.df['target']
        self.df['target'] = self.df.target_word.apply(self.vocab.str2id)
        self.length = len(self.df)
        self.df = self.df.reset_index()
        self.df = self.df.to_dict('index')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        data_line = self.df[item]
        target = data_line['target']
        return t.LongTensor(target), len(target)


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        targets = [i[0] for i in batch]
        targets_lengths = t.LongTensor([i[1] for i in batch])
        padded_target = pad_sequence(targets, batch_first=True)
        return padded_target, targets_lengths


def load_manifests(manifest_list):
    df = pd.DataFrame()
    for i in manifest_list:
        ndf = pd.read_csv(i)
        df = pd.concat([df, ndf])
    return df


def build_raw_data_loader_lm(manifest_list, vocab_path, batch_size, num_workers):
    dataset = LMDataSet(manifest_list, vocab_path=vocab_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=CollateFn(),
                            drop_last=True, shuffle=True)
    return dataloader
