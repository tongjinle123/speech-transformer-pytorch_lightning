from src.bak.data_loader.featurizer.utils import load
from src.bak.data_loader.featurizer.utils import Fbank
from src.bak.data_loader.featurizer.utils import normalization
from src.bak.data_loader.featurizer.utils import concat_and_subsample
from src.bak.data_loader.featurizer.utils import speed_perturb
from src.utils.vocab import Vocab
import torch as t


class Featurizer:
    def __init__(self, n_mel=80, left_frames=3, right_frames=0, skip_frames=2, vocab_path=None, speed_perturb=False):
        super(Featurizer, self).__init__()
        self.fbank = Fbank(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=n_mel)
        # self.fbank = MelSpectrogram(
        #     sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)
        if not vocab_path is None:
            self.vocab = Vocab(vocab_path)
        else:
            self.vocab = None
        self.speed_perturb = speed_perturb

    @property
    def unk_id(self):
        return self.vocab.unk_id

    def __call__(self, file, target=None):
        sig = load(file, do_vad=False)
        if self.speed_perturb:
            sig = speed_perturb(sig, 90, 110, 4)
        # feature = self.fbank(sig)[0].transpose(0, 1).detach()
        # feature = t.log(feature + 1e-10)
        feature = self.fbank(sig)
        feature = normalization(feature)
        feature = concat_and_subsample(feature.numpy(), left_frames=4, skip_frames=3)
        feature_length = len(feature)
        if not self.vocab is None:
            target_id = self.vocab.str2id(target)
            target_length = len(target_id)
        else:
            target_id = None
            target_length = None
        return t.from_numpy(feature), feature_length, t.LongTensor(target_id), target_length



