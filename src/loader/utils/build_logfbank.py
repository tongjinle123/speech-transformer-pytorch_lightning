import librosa
import torch as t
import numpy as np


def remove_empty_line_2d(tensor, empty_value=0):
    """
    to deal with log feature all-zero time step
    :param tensor:
    :param empty_value:
    :return:
    """
    if isinstance(tensor, np.ndarray):
        tensor = t.from_numpy(tensor)
    sequence_length, feature_size = tensor.size()
    mask = tensor.sum(1).ne(empty_value).unsqueeze(1).repeat(1, feature_size)
    tensor = tensor.masked_select(mask).view(-1, feature_size)
    return tensor


def build_logfbank_normalize(sig, n_mels=80, n_fft=512, win_length=400, hop_length=160):
    feature = librosa.feature.melspectrogram(sig, 16000, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, fmin=20.0)
    feature = feature.T
    feature = remove_empty_line_2d(feature)
    if isinstance(feature, np.ndarray):
        feature = t.from_numpy(feature)
    feature = t.log(np.finfo(float).eps + feature)
    feature = (feature - feature.mean()) / feature.std()
    return feature.numpy()


