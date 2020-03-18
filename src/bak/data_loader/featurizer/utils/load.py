import torchaudio as ta
from pyvad import trim
import torch as t


def load(file, do_vad=True):
    sig, sr = ta.load(file, channels_first=True, normalization=True)
    assert sr == 16000
    return sig

