import torchaudio as ta
import torch as t
#
# def fbank(sig, n_mel):
#     feature = ta.compliance.kaldi.fbank(sig, num_mel_bins=n_mel)
#     return feature


class Fbank:
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80):
        self.fbank = ta.transforms.MelSpectrogram(sample_rate, n_fft, win_length, hop_length, n_mels=n_mels)

    def __call__(self, sig):
        return t.log(self.fbank(sig) + 1e-20).squeeze(0).transpose(0, 1).detach()

