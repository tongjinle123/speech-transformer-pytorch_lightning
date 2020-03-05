import torch as t
from src.model.modules.token_encoder import TokenEncoder
from src.utils.vocab import Vocab
from src.utils.masker import Masker
from src.utils.label_smoothing_ce_loss import LabelSmoothingLoss


class TransformerLM(t.nn.Module):
    def __init__(self, embedding_size=512, feed_forward_size=1024, hidden_size=64, dropout=0.1, num_head=8, num_layer=6,
                 vocab_path='testing_vocab_2.model', max_length=100, share_weight=True, use_low_rank=False):
        super(TransformerLM, self).__init__()
        self.vocab = Vocab(vocab_path)
        self.token_encoder = TokenEncoder(
            input_size=embedding_size,
            feed_forward_size=feed_forward_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_head=num_head,
            num_layer=num_layer,
            max_length=max_length,
            share_weight=share_weight,
            vocab_size=self.vocab.vocab_size,
            padding_idx=self.vocab.pad_id,
            use_low_rank=use_low_rank
        )
        self.ce_loss = LabelSmoothingLoss(
            size=self.vocab.vocab_size, smoothing=0.0, padding_idx=self.vocab.pad_id)

    def build_sample_data(self, cuda=False):
        target = t.LongTensor([[1, 2, 3, 4, 5, 6, 7], [1,2,3,4,5,6,0]])
        target_length = t.LongTensor([7, 6])
        if cuda:
            return target.cuda(), target_length.cuda()
        else:
            return target, target_length

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

    def forward(self, ori_token, ori_token_length, cal_ce_loss=True):
        input_token, output_token, token_length, token_mask, token_self_attention_mask = self._prepare_token(
            ori_token, ori_token_length)
        logit = self.token_encoder(input_token, token_mask, token_self_attention_mask)
        if not cal_ce_loss:
            ce_loss = None
        else:
            ce_loss = self.cal_ce_loss(logit, output_token)
        return logit, output_token, token_length, ce_loss

    def get_last_prob(self, ori_token, ori_token_length):
        input_token, output_token, token_length, token_mask, token_self_attention_mask = self._prepare_token(
            ori_token, ori_token_length)
        logit = self.token_encoder(input_token, token_mask, token_self_attention_mask)
        prob = t.nn.functional.log_softmax(logit, -1)
        return prob[:, -1, :]

    def cal_ce_loss(self, logit, output_token):
        ce_loss = t.nn.functional.cross_entropy(
            logit.transpose(-1, -2), output_token, ignore_index=self.vocab.pad_id)
        return ce_loss

    def cal_perplexity(self, logit, output_token):
        perplexity = None
        return perplexity

