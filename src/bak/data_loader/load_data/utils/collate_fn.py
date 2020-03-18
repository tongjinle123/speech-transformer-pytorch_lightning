import torch as t
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        middle = list(map(
            lambda x: {
                'feature': t.from_numpy(np.frombuffer(x['feature'], dtype='float32').reshape(x['feature_shape'])),
                'feature_length': t.from_numpy(x['feature_length'])[0],
                'target': t.from_numpy(x['target']),
                'target_length': t.from_numpy(x['target_length'])
            },
            batch
        ))
        feature = pad_sequence([i['feature'] for i in middle], batch_first=True, padding_value=0.0)
        feature_length = t.stack([i['feature_length'] for i in middle])
        target = pad_sequence([i['target'] for i in middle], batch_first=True, padding_value=0.0)
        target_length = t.stack([i['target_length'] for i in middle])
        return feature, feature_length, target, target_length

