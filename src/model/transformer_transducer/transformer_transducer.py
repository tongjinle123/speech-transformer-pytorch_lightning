import torch as t
from src.model.modules.spec_augment_layer import SpecAugment
from src.utils.masker import Masker
from src.utils.vocab import Vocab
from src.model.modules.spec_encoder import SpecEncoder
from src.model.transformer_transducer.language_model import TransformerLanguageModel
from src.model.transformer_transducer.join_net import JoinNet

from warp_rnnt import rnnt_loss
from warp_rna import rna_loss


class TransformerTransducer(t.nn.Module):
    def __init__(self, num_time_mask=2, num_freq_mask=2, freq_mask_length=15, time_mask_length=15, feature_dim=320,
                 model_size=512, feed_forward_size=1024, hidden_size=64, dropout=0.1, num_head=8, num_layer=8,
                 vocab_path='testing_vocab.model', max_feature_length=1024, max_token_length=50, enable_spec_augment=True):
        super(TransformerTransducer, self).__init__()
        self.enable_spec_augment = enable_spec_augment
        self.max_token_length = max_token_length
        self.vocab = Vocab(vocab_path)
        if enable_spec_augment:
            self.spec_augment = SpecAugment(
                num_time_mask=num_time_mask, num_freq_mask=num_freq_mask,
                freq_mask_length=freq_mask_length, time_mask_length=time_mask_length,
                max_sequence_length=max_feature_length)
        self.input_linear = t.nn.Linear(feature_dim, model_size, bias=True)
        self.spec_encoder = SpecEncoder(
            input_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size, dropout=dropout,
            num_head=num_head, num_layer=num_layer, padding_idx=self.vocab.pad_id, init_size=max_feature_length)
        self.language_model = TransformerLanguageModel(
            input_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size, dropout=dropout,
            num_head=num_head, num_layer=num_layer, vocab_size=self.vocab.vocab_size, padding_idx=self.vocab.pad_id,
            max_length=max_token_length)
        self.join_net = JoinNet(
            encoder_dim=model_size, lm_dim=model_size, model_size=model_size, vocab_size=self.vocab.vocab_size)
        # t.nn.init.xavier_normal_(self.input_linear.weight)

    def build_sample_data(self, feature_dim=320, cuda=False):
        feature = t.randn((32, 120, feature_dim))
        feature_length = t.LongTensor([i for i in range(121-32, 121)])
        target = t.LongTensor([[1, 2, 3, 4, 5, 6, 7]]*32)
        target_length = t.LongTensor([7]*32)
        if cuda:
            return feature.cuda(), feature_length.cuda(), target.cuda(), target_length.cuda()
        else:
            return feature, feature_length, target, target_length

    def _prepare_feature(self, feature, feature_length):
        if self.enable_spec_augment:
            feature = self.spec_augment(feature, feature_length)
        feature_mask = Masker.get_mask(feature_length)
        self_attention_mask = Masker.get_dot_mask(feature_mask, feature_mask)
        return feature, feature_mask, self_attention_mask

    def _prepare_token(self, token, token_length):
        input_token, output_token, token_length = self._rebuild_target(token, token_length)
        token_mask = Masker.get_mask(token_length)
        token_self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
        token_self_attention_mask = Masker.get_forward_mask(token_self_attention_mask)
        return input_token, output_token, token_length, token_mask, token_self_attention_mask

    def _rebuild_target(self, target, target_length):
        input_ = t.nn.functional.pad(target, (1, 0), value=self.vocab.bos_id)
        target_ = t.nn.functional.pad(target, (0, 1), value=self.vocab.pad_id)
        indices = t.LongTensor([[i, v.item()] for i, v in enumerate(target_length)]).to(target.device)
        values = t.LongTensor([self.vocab.eos_id for i in target_length]).to(target.device)
        target_ = target_.index_put(tuple(indices.t()), values=values)
        return input_.detach(), target_.detach(), target_length+1

    def forward(self, feature, feature_length, ori_token, ori_token_length, cal_rnnt_loss=False, cal_cer=False):
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(feature, feature_length)
        input_token, output_token, token_length, token_mask, token_self_attention_mask = self._prepare_token(ori_token, ori_token_length)
        feature = self.input_linear(feature)
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)
        token_feature = self.language_model(input_token, token_self_attention_mask)
        joint = self.join_net(spec_feature, token_feature)
        if cal_rnnt_loss:
            rnn_t_loss = self.cal_transducer_loss(joint, ori_token, feature_length, ori_token_length)
            return joint, rnn_t_loss
        else:
            return joint

    def encode_spec_feature(self, feature, feature_length):
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(feature, feature_length)
        feature = self.input_linear(feature)
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)
        return spec_feature

    def encode_token_id(self, token_id, token_length):
        pass

    def train_ctc_step(self, batch, batch_nb):
        pass

    def train_lm_step(self, batch, batch_nb):
        pass

    def joint_training_step(self, batch, batch_nb):
        pass

    def cal_ctc_loss(self):
        pass

    def cal_transducer_loss(self, model_output, target, frame_length, target_length, type='rnnt'):
        log_prob = t.nn.functional.log_softmax(model_output, -1)
        rnn_t_loss = rnnt_loss(
            log_probs=log_prob, labels=target.int(), frames_lengths=frame_length.int(), labels_lengths=target_length.int(), reduction='mean')
        return rnn_t_loss

    def cal_ce_loss(self):
        pass

    def cal_cer(self):
        pass



