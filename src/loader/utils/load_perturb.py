import librosa
import soundfile
import torch as t


def load_perturb(file, rate=1.0):
    sig, _ = soundfile.read(file, dtype='float32')
    if rate != 1.0:
        sig = speed_perturb2(t.from_numpy(sig).unsqueeze(0), given_rate=rate).numpy().squeeze(0)

    return sig


def speed_perturb2(sig, given_rate=None):
    rate = given_rate
    perturbed = t.nn.functional.interpolate(sig.unsqueeze(0), scale_factor=rate)
    return perturbed[0]