import torch as t
from torch.utils.data import Dataset, DataLoader
from src.loader.utils.load_file_main import LoadFile
from src.utils.vocab import Vocab
import pandas as pd
import os


class DumpedAudioSet(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = [os.path.join(self.root, i) for i in os.listdir(root)]
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        line = t.load(self.files[item])
        return line[0], line[1], line[2], line[3]


class AudioSet(Dataset):
    def __init__(self, manifest, rate_min=0.9, rate_max=1.1, n_mels=80, hop_length=160, win_length=400, n_fft=512,
                 left_frames=3, right_frames=0, skip_frames=2, vocab_path='testing_vocab.model', min_duration=1,
                 max_duration=10, given_rate=None):
        super(AudioSet, self).__init__()
        self.given_rate = given_rate
        self.load_file = LoadFile(
            rate_min, rate_max, n_mels, n_fft, win_length, hop_length, left_frames, right_frames, skip_frames)
        self.vocab = Vocab(vocab_path)
        self.load_manifest(manifest, min_duration, max_duration)

    def load_manifest(self, manifest, min_duration, max_duration):
        self.df = pd.read_csv(manifest)
        self.df = self.df[(self.df.duration > min_duration) & (self.df.duration < max_duration)]
        former = len(self.df)
        self.df = self.df[~self.df.target.apply(lambda x: self.vocab.str2id(x)).apply(
            lambda x: self.vocab.unk_id in x)]
        print(
            f'filtered {manifest}: {former - len(self.df)} datas, former is {former}, now {len(self.df)}')

        self.df = self.df.reset_index()
        self.df = self.df[['wav_file', 'target']]
        self.df = self.df.to_dict('index')
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        data_line = self.df[item]
        file, target = data_line['wav_file'], data_line['target']
        feature = self.load_file(file, self.given_rate)
        feature_length = feature.shape[0]
        target = self.vocab.str2id(target)
        target_length = len(target)
        return feature, feature_length, target, target_length

    def load_wav(self, file):
        feature = self.load_file(file, self.given_rate)
        feature_length = feature.shape[0]
        return feature, feature_length


if __name__ == '__main__':
    audioset = AudioSet('data/manifest/ce_200.csv')
    next(iter(audioset))