import torchaudio as ta
from pyvad import trim
import torch as t


def load(file, do_vad=True):
    sig, sr = ta.load(file, channels_first=True, normalization=True)
    assert sr == 16000
    if do_vad:
        trimed = trim(sig.transpose(0, 1).numpy(), fs=sr, fs_vad=16000, hoplength=30, vad_mode=2)
        if trimed is not None:
            return t.from_numpy(trimed).transpose(0, 1)
        else:
            return sig
    else:
        return sig

