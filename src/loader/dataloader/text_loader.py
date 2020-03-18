import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from prefetch_generator import BackgroundGenerator
from src_reshaped.loader.dataloader.datasets.text_set import TextSet


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


class CollateFn:

    def __init__(self):
        pass

    def __call__(self, batch):
        targets = [t.LongTensor(i[0]) for i in batch]
        targets_lengths = t.LongTensor([i[1] for i in batch])
        padded_target = pad_sequence(targets, batch_first=True)
        return padded_target, targets_lengths


def build_text_data_loader(manifest_list, batch_size=32, num_workers=16, shuffle=True, drop_last=True,
                           vocab_path='testing_vocab.model'):
    audio_sets = [TextSet(file, vocab_path) for file in manifest_list]
    dataset = ConcatDataset(audio_sets)
    dataloader = DataLoaderX(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=CollateFn(), drop_last=drop_last)
    return dataloader

if __name__ == '__main__':
    loader = build_text_data_loader(['data/manifest/ce_20_dev.csv'])
    a = next(iter(loader))