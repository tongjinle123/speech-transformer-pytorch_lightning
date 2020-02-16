import torch as t


def normalization(feature):
    #feature = t.from_numpy(feature)
    mean = t.mean(feature)
    std = t.std(feature)
    return ((feature - mean) / std)#.numpy()

