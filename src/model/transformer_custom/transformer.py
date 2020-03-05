import torch as t
from src.model.modules.spec_augment_layer import SpecAugment
from src.utils.masker import Masker
from src.utils.vocab import Vocab
from src.model.modules.spec_encoder import SpecEncoder
# from src.model.modules.token_decoder_swich import TokenDecoder
from src.utils.label_smoothing_ce_loss import LabelSmoothingLoss


class Transformer(t.nn.Module):
    def __init__(self, num_time_mask=2, num_freq_mask=2, freq_mask_length=15, time_mask_length=15, feature_dim=320,
                 model_size=512, feed_forward_size=1024, hidden_size=64, dropout=0.1, num_head=8, num_encoder_layer=6,
                 num_decoder_layer=6, vocab_path='testing_vocab_2.model', max_feature_length=1024, max_token_length=50,
                 enable_spec_augment=True, share_weight=True, smoothing=0.1, restrict_left_length=10,
                 restrict_right_length=10, use_low_rank=False):
        super(Transformer, self).__init__()

        self.enable_spec_augment = enable_spec_augment
        self.max_token_length = max_token_length
        self.restrict_left_length = restrict_left_length
        self.restrict_right_length = restrict_right_length
        self.vocab = Vocab(vocab_path)
        if enable_spec_augment:
            self.spec_augment = SpecAugment(
                num_time_mask=num_time_mask, num_freq_mask=num_freq_mask,
                freq_mask_length=freq_mask_length, time_mask_length=time_mask_length,
                max_sequence_length=max_feature_length)
        self.spec_encoder = SpecEncoder(
            input_size=feature_dim, model_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size,
            dropout=dropout, num_head=num_head, num_layer=num_encoder_layer, padding_idx=self.vocab.pad_id,
            init_size=max_feature_length, use_low_rank=use_low_rank)
        self.encoder_linear = t.nn.Linear(model_size, self.vocab.vocab_size, bias=True)
        t.nn.init.xavier_normal_(self.encoder_linear.weight)
        self.encoder_linear_switch = t.nn.Linear(model_size, 4, bias=True)
        t.nn.init.xavier_normal_(self.encoder_linear_switch.weight)
        self.token_decoder = TokenDecoder(
            input_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size, dropout=dropout,
            num_head=num_head, num_layer=num_decoder_layer, vocab_size=self.vocab.vocab_size,
            padding_idx=self.vocab.pad_id, max_length=max_token_length, share_weight=share_weight,
            bos_id=self.vocab.bos_id, eos_id=self.vocab.eos_id, use_low_rank=use_low_rank
        )
        self.switch_loss = LabelSmoothingLoss(size=3, smoothing=0, padding_idx=0)
        self.label_smoothing_celoss = LabelSmoothingLoss(
            self.vocab.vocab_size, smoothing=smoothing, padding_idx=self.vocab.pad_id)

    def build_sample_data(self, feature_dim=320, cuda=False):
        feature = t.randn((2, 120, feature_dim))
        feature_length = t.LongTensor([i for i in range(119, 121)])
        target = t.LongTensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 0]])
        target_length = t.LongTensor([7, 6])
        if cuda:
            return feature.cuda(), feature_length.cuda(), target.cuda(), target_length.cuda()
        else:
            return feature, feature_length, target, target_length

    def _prepare_feature(self, feature, feature_length, restrict_left_length=None, restrict_right_length=None):
        """
        do spec augment and build mask
        """
        if self.enable_spec_augment:
            feature = self.spec_augment(feature, feature_length)
        feature_mask = Masker.get_mask(feature_length)
        self_attention_mask = Masker.get_dot_mask(feature_mask, feature_mask)
        self_attention_mask = Masker.get_restricted_mask(
            self_attention_mask, restrict_left_length, restrict_right_length)
        return feature, feature_mask, self_attention_mask

    def _prepare_token(self, token, token_length):
        """
        build target and mask
        """
        input_token, output_token, token_length = self._rebuild_target(token, token_length)
        token_mask = Masker.get_mask(token_length)
        token_self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
        token_self_attention_mask = Masker.get_forward_mask(token_self_attention_mask)

        switch = t.ones_like(output_token, device=token.device).long()  # eng = 1
        switch.masked_fill_(output_token.eq(0), 0)  # pad=0
        switch.masked_fill_((output_token.ge(12) & output_token.le(4211)), 2)  # ch = 2

        enc_switch = t.ones_like(token, device=token.device).long()  # eng = 1
        enc_switch.masked_fill_(token.eq(0), 0)  # pad=0
        enc_switch.masked_fill_((token.ge(12) & token.le(4211)), 2)  # ch = 2
        return input_token, output_token, token_length, token_mask, token_self_attention_mask, switch, enc_switch

    def _rebuild_target(self, target, target_length):
        """
        add eos & bos into original token in a batched tensor way
        """
        input_ = t.nn.functional.pad(target, (1, 0), value=self.vocab.bos_id)
        target_ = t.nn.functional.pad(target, (0, 1), value=self.vocab.pad_id)
        indices = t.LongTensor([[i, v.item()] for i, v in enumerate(target_length)]).to(target.device)
        values = t.LongTensor([self.vocab.eos_id for i in target_length]).to(target.device)
        target_ = target_.index_put(tuple(indices.t()), values=values)
        return input_, target_, target_length + 1

    def forward(self, feature, feature_length, ori_token, ori_token_length, cal_ce_loss=True):
        #
        t.cuda.empty_cache()
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(
            feature, feature_length, restrict_left_length=self.restrict_left_length,
            restrict_right_length=self.restrict_right_length)
        #
        input_token, output_token, token_length, token_mask, token_self_attention_mask, swich_target, enc_switch_target=self._prepare_token(
            ori_token, ori_token_length)
        #
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)
        #
        spec_output = self.encoder_linear(spec_feature)
        #
        spec_swtich_out = self.encoder_linear_switch(spec_feature)
        #
        dot_attention_mask = Masker.get_dot_mask(token_mask, feature_mask)
        output, swich = self.token_decoder(
            input_token, spec_feature, token_mask, token_self_attention_mask, dot_attention_mask)
        switch_loss = self.switch_loss(swich, swich_target)
        if cal_ce_loss:
            ce_loss = self.cal_ce_loss(output, output_token, type='lbce')
        else:
            ce_loss = None
        return output, output_token, spec_output, feature_length, ori_token, ori_token_length, ce_loss, switch_loss, spec_swtich_out, enc_switch_target

    def cal_ce_loss(self, decoder_output, output_token, type='ce'):
        if type == 'lbce':
            ce_loss = self.label_smoothing_celoss(decoder_output, output_token)
        elif type == 'ce':
            ce_loss = t.nn.functional.cross_entropy(
                decoder_output.transpose(-1, -2), output_token, ignore_index=self.vocab.pad_id)
        return ce_loss

    def cal_ctc_loss(self, feature_prob, feature_length, token, token_length):
        prob = t.nn.functional.log_softmax(feature_prob, -1)
        ctc_loss = t.nn.functional.ctc_loss(
            prob.transpose(0, 1), token, feature_length, token_length, blank=self.vocab.blank_id, zero_infinity=True)
        return ctc_loss

    def cal_ctc_switch_loss(self, feature_prob, feature_length, token, token_length):
        prob = t.nn.functional.log_softmax(feature_prob, -1)
        ctc_loss = t.nn.functional.ctc_loss(
            prob.transpose(0, 1), token, feature_length, token_length, blank=3, zero_infinity=True
        )
        return ctc_loss


    def decode(self, feature, feature_length):
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(
            feature, feature_length, restrict_left_length=20, restrict_right_length=20)
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)
        spec_output = self.encoder_linear(spec_feature)



        # dot_attention_mask = Masker.get_dot_mask(token_mask, feature_mask)
        # output, swich = self.token_decoder(
        #     input_token, spec_feature, token_mask, token_self_attention_mask, dot_attention_mask)
        #

    def inference(self, feature, feature_length, decode_type='greedy'):
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(
            feature, feature_length, restrict_left_length=20, restrict_right_length=20)
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)

        def cut_sentence(line_token, eos_id):
            nline_token = []
            for i in line_token.tolist():
                if i != eos_id:
                    nline_token.append(i)
                else:
                    break
            return nline_token

        if decode_type == 'greedy':
            decoded_token = self.token_decoder.greedy_decode(
                encoder_output=spec_feature, feature_mask=feature_mask)
            decoded_string = [self.vocab.id2string(cut_sentence(i, self.vocab.eos_id)) for i in decoded_token]
            return decoded_string

        elif decode_type == 'beam':
            decoded_token = self.token_decoder.beam_search_decode(
                encoder_output=spec_feature, feature_mask=feature_mask, beam_size=5)
            return decoded_token
            # print(decoded_token)
            # decoded_string = [self.vocab.id2string(cut_sentence(i, self.vocab.eos_id)) for i in decoded_token]
            # return decoded_string


import torch as t
from src.model.modules.transformer_decoder import TransformerDecoder
from src.model.modules.embedding import Embedding
from src.utils.masker import Masker
from src.model.modules.beam_searcher import BeamSteper


class TokenDecoder(t.nn.Module):
    """
    token transformer decoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, vocab_size,
                 padding_idx, max_length=50, share_weight=True, bos_id=3, eos_id=4, use_low_rank=False):
        super(TokenDecoder, self).__init__()
        self.max_length = max_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.embedding = Embedding(
            vocab_size, input_size, padding_idx, max_length, scale_word_embedding=share_weight)
        self.transformer_decoder = TransformerDecoder(
            input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1/(input_size ** -0.5))
        self.output_layer = t.nn.Linear(input_size, vocab_size, bias=False)
        self.switch_layer = t.nn.Linear(input_size, 3, bias=False)
        t.nn.init.xavier_normal_(self.switch_layer.weight)
        if share_weight:
            self.output_layer.weight = self.embedding.word_embedding.weight
        else:
            t.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
        net = self.embedding(token_id)
        net.masked_fill_(token_mask.unsqueeze(-1) == 0, 0.0)
        net = self.transformer_decoder(net, token_mask.unsqueeze(-1), encoder_output, self_attention_mask, dot_attention_mask)
        net = self.layer_norm(net)
        swich = self.switch_layer(net)
        net = self.output_layer(net)
        return net, swich

    def beam_search_decode(self, encoder_output, feature_mask, beam_size, best_k=5, lp_eps=0.0):
        batch_size, feature_length, _ = encoder_output.size()
        device = encoder_output.device
        self.beam_steper = BeamSteper(
            batch_size=batch_size, beam_size=beam_size, bos_id=self.bos_id, eos_id=self.eos_id,
            vocab_size=self.vocab_size, device=device, k_best=best_k, lp_eps=lp_eps
        )

        beam_feature_mask = feature_mask.unsqueeze(1).repeat(1, beam_size, 1).view(batch_size*beam_size, -1)
        beam_encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_size, 1, 1).view(batch_size*beam_size, feature_length, -1)
        with t.no_grad():
            for i in range(self.max_length):
                if i == 0:
                    token_id = self.beam_steper.get_first_step_token()
                    length = self.beam_steper.get_first_step_length()
                    token_mask = Masker.get_mask(length)
                    self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
                    self_attention_mask = Masker.get_forward_mask(self_attention_mask)
                    dot_attention_mask = Masker.get_dot_mask(token_mask, feature_mask)
                    last_prob = self.beam_decode_step(
                        token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
                    if_continue = self.beam_steper.first_step(last_prob)
                    if not if_continue:
                        break
                else:
                    token_id = self.beam_steper.token_container
                    token_id = token_id.view(batch_size * beam_size, -1)
                    length = self.beam_steper.length_container
                    length = length.view(batch_size * beam_size)
                    token_mask = Masker.get_mask(length)
                    self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
                    self_attention_mask = Masker.get_forward_mask(self_attention_mask)
                    dot_attention_mask = Masker.get_dot_mask(token_mask, beam_feature_mask)
                    last_prob = self.beam_decode_step(
                        token_id, beam_encoder_output, token_mask, self_attention_mask, dot_attention_mask
                    )
                    if_continue = self.beam_steper.step(last_prob.view(batch_size, beam_size, -1))
                    if not if_continue:
                        break
        output_token = self.beam_steper.batch_best_saver.batch
        return output_token

    def greedy_decode(self, encoder_output, feature_mask):
        """
        batched greedy decode
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        token_id = t.full((batch_size, 1), fill_value=self.bos_id, dtype=t.long, device=device)
        length = t.LongTensor([1] * batch_size).to(device)
        #probs = t.Tensor().to(device)
        with t.no_grad():
            for i in range(self.max_length):
                try:
                    token_mask = Masker.get_mask(length)
                    self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
                    self_attention_mask = Masker.get_forward_mask(self_attention_mask)
                    dot_attention_mask = Masker.get_dot_mask(token_mask, feature_mask)
                    last_prob, last_token_id = self.decode_step(
                        token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask,
                        topk=1, return_last=True)
                    token_id = t.cat([token_id, last_token_id], dim=1)
                    # print('concate, tokenid', token_id)
                    #probs = t.cat([probs, last_prob], dim=1)
                    for index, id in enumerate(last_token_id.squeeze(1)):
                        if id != self.eos_id:
                            length[index] += 1
                except:
                    #TODO: to be more consious
                    break
        return token_id
    #
    # def greedy_decode(self, encoder_output, dot_attention_mask):
    #     """
    #     batched greedy decode
    #     """
    #     batch_size = encoder_output.size(0)
    #     device = encoder_output.device
    #     token_id = t.full((batch_size, 1), fill_value=self.bos_id, dtype=t.long, device=device)
    #     length = t.LongTensor([1] * batch_size).to(device)
    #     #probs = t.Tensor().to(device)
    #     with t.no_grad():
    #         for i in range(self.max_length):
    #             try:
    #                 token_mask = Masker.get_mask(length)
    #                 self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
    #                 last_prob, last_token_id = self.decode_step(
    #                     token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask,
    #                     topk=1, return_last=True)
    #                 token_id = t.cat([token_id, last_token_id], dim=1)
    #                 # print('concate, tokenid', token_id)
    #                 #probs = t.cat([probs, last_prob], dim=1)
    #                 for index, id in enumerate(last_token_id.squeeze(1)):
    #                     if id != self.eos_id:
    #                         length[index] += 1
    #             except:
    #                 #TODO: to be more consious
    #                 break
    #     return token_id

        # B, 1

    def beam_decode_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
        net, _ = self.forward(token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
        net = t.nn.functional.log_softmax(net, -1)
        return net[:, -1, :]

    def decode_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask, topk=1,
                    return_last=True):
        # token_id B, Lt
        # encoder_output B, Lf, H
        # token_mask B, Lt
        # self_attention_mask B, 1, Lt
        # dot_attention_mask B, 1, Lf
        net, _ = self.forward(token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
        net = t.nn.functional.log_softmax(net, -1)
        probs, indexs = t.topk(net, topk)
        if return_last:
            return probs[:, -1, :], indexs[:, -1, :]
        else:
            return probs, indexs

