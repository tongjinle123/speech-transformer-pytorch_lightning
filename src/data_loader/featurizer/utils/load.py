import torchaudio as ta
from pyvad import trim
import torch as t


def load(file, do_vad=True):
    sig, sr = ta.load(file, channels_first=True, normalization=True)
    assert sr == 16000
    if do_vad:
        start, end = trim(sig.transpose(0, 1).numpy(), fs=sr, fs_vad=16000, hop_length=30, vad_mode=2)
        if start != 0 and end != 0:
            return sig[:, start: end]
        else:
            return sig
    else:
        return sig

