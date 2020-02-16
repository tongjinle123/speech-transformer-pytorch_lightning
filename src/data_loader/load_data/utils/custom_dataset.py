from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset
from torch.utils.data import DataLoader

class CustomTFRecordDataset(TFRecordDataset):
    def __init__(self, *args, length):
        super(CustomTFRecordDataset, self).__init__(*args)
        self.length = length

    def __len__(self):
        return self.length


class CustomMultiTFRecordDataset(MultiTFRecordDataset):
    def __init__(self, length, *args, **kwargs):
        super(CustomMultiTFRecordDataset, self).__init__(*args, **kwargs)
        self.length = length

    def __len__(self):
        return self.length


class CustomTFRecordDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomTFRecordDataLoader, self).__init__(*args, **kwargs)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size) - 1


