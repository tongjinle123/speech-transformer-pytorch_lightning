import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from prefetch_generator import BackgroundGenerator
from src.loader.dataloader.datasets.auido_set import AudioSet
from src.loader.dataloader.datasets.auido_set import DumpedAudioSet
import numpy as np


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


class CollateFn:

    def __init__(self):
        pass

    def __call__(self, batch):
        features = [t.from_numpy(i[0]) for i in batch]
        features_lengths = t.LongTensor([i[1] for i in batch])
        targets = [t.LongTensor(i[2]) for i in batch]
        targets_lengths = t.LongTensor([i[3] for i in batch])
        padded_feature = pad_sequence(features, batch_first=True)
        padded_target = pad_sequence(targets, batch_first=True)
        return padded_feature, features_lengths, padded_target, targets_lengths


class CollateFnDump:
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch


def build_data_loader(
        manifest_list, batch_size=32, num_workers=16, shuffle=True, drop_last=True, rate_min=0.9, rate_max=1.1,
        n_mels=80, hop_length=160, win_length=400, n_fft=512, left_frames=3, right_frames=0, skip_frames=2,
        vocab_path='testing_vocab.model', min_duration=1, max_duration=10, given_rate=None):
    audio_sets = [
        AudioSet(
            file, rate_min, rate_max, n_mels, hop_length, win_length, n_fft, left_frames, right_frames,
            skip_frames, vocab_path, min_duration, max_duration, given_rate
        ) for file in manifest_list
    ]
    dataset = ConcatDataset(audio_sets)
    dataloader = DataLoaderX(
        dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=CollateFn(), drop_last=drop_last)
    return dataloader


def build_predumped_loader(root_list, batch_size=32, num_workers=16, shuffle=True, drop_last=True):
    audio_sets = [
        DumpedAudioSet(file) for file in root_list
    ]
    dataset = ConcatDataset(audio_sets)
    dataloader = DataLoaderX(
        dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=CollateFn(), drop_last=drop_last)
    return dataloader


def build_data_loader_dump(
        manifest_list, batch_size=32, num_workers=16, shuffle=True, drop_last=True, rate_min=0.9, rate_max=1.1,
        n_mels=80, hop_length=160, win_length=400, n_fft=512, left_frames=0, right_frames=0, skip_frames=0,
        vocab_path='testing_vocab.model', min_duration=1, max_duration=10, given_rate=None):

    audio_sets = [
        AudioSet(
            file, rate_min, rate_max, n_mels, hop_length, win_length, n_fft, left_frames, right_frames,
            skip_frames, vocab_path, min_duration, max_duration, given_rate
        ) for file in manifest_list
    ]
    dataset = ConcatDataset(audio_sets)
    dataloader = DataLoaderX(
        dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=CollateFnDump(), drop_last=drop_last)
    return dataloader