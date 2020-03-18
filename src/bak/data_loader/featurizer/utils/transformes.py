from __future__ import absolute_import, division, print_function, unicode_literals
from warnings import warn
import math
import torch
from typing import Optional
import torchaudio.functional as F





class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins
        win_length (int): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        pad (int): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[[...], torch.Tensor]): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (int): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...]): Arguments for window function. (Default: ``None``)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window,
                 power=2, normalized=False, wkwargs=None):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.window = window
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (channel, time)

        Returns:
            torch.Tensor: Dimension (channel, freq, time), where channel
            is unchanged, freq is ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frames).
        """
        return F.spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
                             self.win_length, self.power, self.normalized)


class MelScale(torch.nn.Module):
    r"""This turns a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int): Sample rate of audio signal. (Default: ``16000``)
        f_min (float): Minimum frequency. (Default: ``0.``)
        f_max (float, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`.
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, n_mels=128, sample_rate=16000, f_min=0., f_max=None, n_stft=None):
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)
        self.f_min = f_min
        fb = torch.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels)
        self.fb = fb

    def forward(self, specgram):
        r"""
        Args:
            specgram (torch.Tensor): A spectrogram STFT of dimension (channel, freq, time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (channel, ``n_mels``, time)
        """
        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)
        return mel_specgram


class MelSpectrogram(torch.nn.Module):
    r"""Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
    and MelScale.

    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: ``16000``)
        win_length (int): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins
        f_min (float): Minimum frequency. (Default: ``0.``)
        f_max (float, optional): Maximum frequency. (Default: ``None``)
        pad (int): Two sided padding of signal. (Default: ``0``)
        n_mels (int): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[[...], torch.Tensor]): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        wkwargs (Dict[..., ...]): Arguments for window function. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalization=True)
        >>> mel_specgram = transforms.MelSpectrogram(sample_rate)(waveform)  # (channel, n_mels, time)
    """
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0., f_max=None,
                 pad=0, n_mels=128, window_fn=torch.hann_window, wkwargs=None):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=2,
                                       normalized=False, wkwargs=wkwargs)
        self.mel_scale = MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max)

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (channel, time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (channel, ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


