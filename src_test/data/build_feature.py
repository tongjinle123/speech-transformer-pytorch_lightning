import soundfile
import torchaudio as ta
import torch as t
from src.utils.vocab import Vocab


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


def load_file(file, left_frames=3, right_frames=0, skip_frames=2):
    sig, _ = soundfile.read(file, dtype='int16')
    sig = t.from_numpy(sig).unsqueeze(0)
    feature = ta.compliance.kaldi.fbank(sig, num_mel_bins=80,use_log_fbank=True)
    feature = (feature - feature.mean()) / feature.std()
    feature = concat_and_subsample(feature, left_frames=left_frames, right_frames=right_frames, skip_frames=skip_frames)
    return feature


class GetBatch:
    def __init__(self, vocab='testing_vocab.model'):
        self.vocab = Vocab(vocab)

