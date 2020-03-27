from .build_logfbank import build_logfbank_normalize
from .concat_and_subsample import concat_and_subsample
from .load_perturb import load_perturb
import random


def load_file(file, rate, n_mels=80, n_fft=512, win_length=400, hop_length=160, left_frames=3,
              right_frames=0, skip_frames=2):
    sig = load_perturb(file, rate)
    feature = build_logfbank_normalize(sig, n_mels, n_fft, win_length, hop_length)
    feature = concat_and_subsample(feature, left_frames, right_frames, skip_frames)
    return feature


class LoadFile:
    def __init__(self, rate_min=0.9, rate_max=0.1, n_mels=80, n_fft=512, win_length=400, hop_length=160,
                 left_frames=3, right_frames=0, skip_frames=2):
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.skip_frames = skip_frames

    def __call__(self, file, given_rate=None):
        if given_rate is not None:
            feature = load_file(
                file, rate=given_rate, n_mels=self.n_mels, n_fft=self.n_fft, win_length=self.win_length,
                hop_length=self.hop_length, left_frames=self.left_frames, right_frames=self.right_frames,
                skip_frames=self.skip_frames)
        else:
            rate = random.randrange(int(self.rate_min * 100), int(self.rate_max * 100)) / 100
            feature = load_file(
                file, rate=rate, n_mels=self.n_mels, n_fft=self.n_fft, win_length=self.win_length,
                hop_length=self.hop_length, left_frames=self.left_frames, right_frames=self.right_frames,
                skip_frames=self.skip_frames)
        return feature