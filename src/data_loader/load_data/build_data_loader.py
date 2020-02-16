from src.data_loader.load_data.utils.custom_dataset import CustomMultiTFRecordDataset, CustomTFRecordDataset, CustomTFRecordDataLoader
import os
import torch as t
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from prefetch_generator import BackgroundGenerator


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        middle = list(map(
            lambda x: {
                'feature': t.from_numpy(np.frombuffer(x['feature'], dtype='float32').reshape(x['feature_shape'])),
                'feature_length': t.from_numpy(x['feature_shape'])[0],
                'target_id': t.from_numpy(x['target_id']),
                'target_length': t.from_numpy(x['target_length'])
            },
            batch
        ))
        feature = pad_sequence([i['feature'] for i in middle], batch_first=True, padding_value=0.0)
        feature_length = t.stack([i['feature_length'] for i in middle])
        target = pad_sequence([i['target_id'] for i in middle], batch_first=True, padding_value=0.0)
        target_length = t.stack([i['target_length'] for i in middle])
        return feature, feature_length.long(), target.long(), target_length.long()


def build_single_dataset(data_path, index_path):
    description = {
        'feature': 'byte',
        'feature_shape': 'int',
        'target_id': 'int',
        'target_length': 'int',
    }
    length = int(data_path.split('/')[-1].split('.')[0].split('_')[-1])
    dataset = CustomTFRecordDataset(data_path,index_path, description,length=length)
    return dataset



def build_single_dataloader(data_path, index_path, batch_size, num_workers):
    dataset = build_single_dataset(data_path, index_path)
    dataloader = CustomTFRecordDataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=CollateFn())
    return dataloader



def build_multi_dataset(record_root, index_root, data_name_list):
    assert sum([1 for i in data_name_list if os.path.exists(os.path.join(record_root.split('{}')[0], i+'.tfrecord'))]) == len(data_name_list)
    splits = {i: int(i.split('_')[-1]) for i in data_name_list}
    total = sum([v for i, v in splits.items()])
    splits = {i: v / total for i, v in splits.items()}
    description = {
        'feature': 'byte',
        'feature_shape': 'int',
        'target_id': 'int',
        'target_length': 'int',
    }
    dataset = CustomMultiTFRecordDataset(
        length=total, data_pattern=record_root, index_pattern=index_root, splits=splits, description=description)
    return dataset


def build_multi_dataloader(record_root, index_root, data_name_list, batch_size, num_workers):
    dataset = build_multi_dataset(record_root, index_root, data_name_list)
    dataloader = CustomTFRecordDataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=CollateFn())
    return dataloader


