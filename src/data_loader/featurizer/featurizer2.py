import soundfile
import torchaudio as ta
import torch as t
from src.utils.vocab import Vocab
import numpy as np


class Featurizer:
    def __init__(self, n_mel=80, left_frames=3, right_frames=0, skip_frames=2, vocab_path=None, speed_perturb=False):
        super(Featurizer, self).__init__()

        if not vocab_path is None:
            self.vocab = Vocab(vocab_path)
        else:
            self.vocab = None
        self.speed_perturb = speed_perturb
        print('using new')

    @property
    def unk_id(self):
        return self.vocab.unk_id

    def __call__(self, file, target=None):

        feature = load_file(file)
        feature_length = feature.shape[0]
        if not self.vocab is None:
            target_id = self.vocab.str2id(target)
            target_length = len(target_id)
        else:
            target_id = None
            target_length = None
        return t.from_numpy(feature), feature_length, t.LongTensor(target_id), target_length


def concat_and_subsample(features, left_frames=3, right_frames=0, skip_frames=2):

    time_steps, feature_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, (1+left_frames+right_frames) * feature_dim], dtype=np.float32)

    concated_features[:, left_frames * feature_dim: (left_frames+1)*feature_dim] = features

    for i in range(left_frames):
        concated_features[i+1: time_steps, (left_frames-i-1)*feature_dim: (
            left_frames-i) * feature_dim] = features[0:time_steps-i-1, :]

    for i in range(right_frames):
        concated_features[0:time_steps-i-1, (right_frames+i+1)*feature_dim: (
            right_frames+i+2)*feature_dim] = features[i+1: time_steps, :]

    return concated_features[::skip_frames+1, :]


def load_file(file, left_frames=4, right_frames=0, skip_frames=3):
    sig, _ = soundfile.read(file, dtype='int16')
    sig = t.from_numpy(sig).unsqueeze(0)
    feature = ta.compliance.kaldi.fbank(sig, num_mel_bins=80, use_log_fbank=True)
    feature = (feature - feature.mean()) / feature.std()
    feature = concat_and_subsample(feature, left_frames=left_frames, right_frames=right_frames, skip_frames=skip_frames)
    return feature
