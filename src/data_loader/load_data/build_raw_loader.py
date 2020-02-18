import torch as t
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.data_loader.featurizer.featurizer import Featurizer
import gc
from prefetch_generator import BackgroundGenerator
from torch.nn.utils.rnn import pad_sequence


class AudioSet(Dataset):
    def __init__(self, manifest_files, max_duration=10, min_duration=1,
                 vocab_path='testing_vocab.model', speed_perturb=False):
        super(AudioSet, self).__init__()
        self.manifest_files = manifest_files
        self.df = pd.DataFrame()
        for mani in manifest_files:
            ndf = pd.read_csv(mani)
            self.df = pd.concat([self.df, ndf])
        del ndf
        self.df = self.df.reset_index()
        self.df = self.df[(self.df.duration < max_duration) & (self.df.duration > min_duration)]
        self.df = self.df[['wav_file', 'target']]

        self.featurizer = Featurizer(speed_perturb=speed_perturb, vocab_path=vocab_path)
        self.filter_unk()
        self.df = self.df.reset_index()
        self.df = self.df[['wav_file', 'target']]
        self.df = self.df.to_dict('index')
        gc.collect()
        self.length = len(self.df)

    def filter_unk(self):
        former = len(self.df)
        self.df = self.df[~self.df.target.apply(lambda x: self.featurizer.vocab.str2id(x)).apply(lambda x:self.featurizer.unk_id in x)]
        print(f'filtered {self.manifest_files}: {former - len(self.df)} datas, former is {former}, now {len(self.df)}')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        data_line = self.df[item]
        file, target = data_line['wav_file'], data_line['target']
        feature, feature_length, target, target_length = self.featurizer(file, target)
        return feature, feature_length, target, target_length

class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        features = [i[0] for i in batch]
        features_lengths = t.LongTensor([i[1] for i in batch])
        targets = [i[2] for i in batch]
        targets_lengths = t.LongTensor([i[3] for i in batch])
        padded_feature = pad_sequence(features, batch_first=True)
        padded_target = pad_sequence(targets, batch_first=True)
        return padded_feature, features_lengths, padded_target, targets_lengths


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


def build_raw_data_loader(manifest_list, vocab_path, batch_size, num_workers, speed_perturb):
    dataset = AudioSet(manifest_list, vocab_path=vocab_path, speed_perturb=speed_perturb)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=CollateFn(), drop_last=True, shuffle=True)
    return dataloader