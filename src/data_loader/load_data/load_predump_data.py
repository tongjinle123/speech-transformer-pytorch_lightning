import torch as t
import os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
from prefetch_generator import BackgroundGenerator
import librosa

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


class FeatureSet(Dataset):
    def __init__(self, root_list):
        self.files = []
        for i in root_list:
            self.files.extend([os.path.join(i, j) for j in os.listdir(i)])
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        a = time.time()
        data = t.load(self.files[item])
        feature, feature_length, target_id, input_target_id, output_target_id, target_length = data['feature'], data[
            'feature_length'], data['target_id'], data['input_target_id'], data['output_target_id'], data[
                                                                                                   'target_length']
        return feature, feature_length, target_id, input_target_id, output_target_id, target_length


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        features = t.stack([i[0] for i in batch], 0)
        features_lengths = t.stack([i[1] for i in batch], 0)
        target_length = t.stack([i[5] for i in batch], 0)
        target_id = [i[2] for i in batch]
        #         input_target_id = [i[3] for i in batch]
        #         output_target_id = [i[4] for i in batch]
        padded_target_id = pad_sequence(target_id, batch_first=True)
        #         padded_input_target_id = pad_sequence(input_target_id, batch_first=True)
        #         padded_output_target_id = pad_sequence(output_target_id, batch_first=True)
        padded_feature = features[:max(features_lengths).item(), :]
        #         return padded_feature, features_lengths, padded_target_id, padded_input_target_id, padded_output_target_id, target_length

        return padded_feature, features_lengths, padded_target_id, target_length


def build_raw_data_loader(root_list, batch_size, num_workers):
    dataset = FeatureSet(root_list)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=CollateFn(), drop_last=True, shuffle=True)
    return dataloader

