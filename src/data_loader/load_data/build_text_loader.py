import torch as t
import pandas as pd
from src.utils.vocab import Vocab
from torch.utils.data import DataLoader, Dataset
import gc
from torch.nn.utils.rnn import pad_sequence


class TextSet(Dataset):
    def __init__(self, manifest_files, vocab_path):
        super(TextSet, self).__init__()
        self.vocab = Vocab(vocab_path)
        self.manifest_files = manifest_files
        self.df = pd.DataFrame()
        for mani in manifest_files:
            ndf = pd.read_csv(mani)
            self.df = pd.concat([self.df, ndf])
        del ndf
        self.df = self.df.reset_index()

        self.df = self.df[['target']]
        self.filter_unk()
        self.df = self.df.reset_index()
        self.df = self.df[['target']]
        self.df = self.df.to_dict('index')
        gc.collect()
        self.length = len(self.df)

    def filter_unk(self):
        former = len(self.df)
        self.df = self.df[~self.df.target.apply(lambda x: self.vocab.str2id(x)).apply(lambda x:self.vocab.unk_id in x)]
        print(f'filtered {self.manifest_files}: {former - len(self.df)} datas, former is {former}, now {len(self.df)}')


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        data_line = self.df[item]
        target = data_line['target']
        target_id = self.vocab.str2id(target)
        target_length = len(target_id)
        return t.LongTensor(target_id), target_length


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        targets = [i[0] for i in batch]
        targets_lengths = t.LongTensor([i[1] for i in batch])
        padded_target = pad_sequence(targets, batch_first=True)
        return padded_target, targets_lengths


def build_text_data_loader(manifest_list, vocab_path, batch_size, num_workers):
    dataset = TextSet(manifest_list, vocab_path=vocab_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=CollateFn(), drop_last=True, shuffle=True)
    return dataloader