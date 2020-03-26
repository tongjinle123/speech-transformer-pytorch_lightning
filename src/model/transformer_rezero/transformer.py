import torch as t
from src.model.modules.spec_augment_layer import SpecAugment
from src.utils.masker import Masker
from src.utils.vocab import Vocab
from src.model.transformer_rezero.customized_modules.rezero_spec_encoder import SpecEncoder
from src.model.transformer_rezero.customized_modules.rezero_token_decoder_switch import TokenDecoder
from src.utils.label_smoothing_ce_loss import LabelSmoothingLoss
from src.utils.ctc_prefix import CTCPrefixScore
from src.utils.end_detect import end_detect
import numpy as np

CTC_SCORING_RATIO = 1.5

class Transformer(t.nn.Module):
    def __init__(self, num_time_mask=2, num_freq_mask=2, freq_mask_length=15, time_mask_length=15, feature_dim=320,
                 model_size=512, feed_forward_size=1024, hidden_size=64, dropout=0.1, num_head=8, num_encoder_layer=6,
                 num_decoder_layer=6, vocab_path='testing_vocab_2.model', max_feature_length=1024, max_token_length=50,
                 enable_spec_augment=True, share_weight=True, smoothing=0.1, restrict_left_length=20,
                 restrict_right_length=20, input_type='linear', use_low_rank=False):
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
            dropout=dropout, num_head=num_head, num_layer=num_encoder_layer, input_type=input_type, use_low_rank=use_low_rank)
        self.encoder_linear = t.nn.Linear(model_size, self.vocab.vocab_size, bias=True)
        t.nn.init.xavier_normal_(self.encoder_linear.weight)
        self.token_decoder = TokenDecoder(
            input_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size, dropout=dropout,
            num_head=num_head, num_layer=num_decoder_layer, vocab_size=self.vocab.vocab_size,
            padding_idx=self.vocab.pad_id,
            max_length=max_token_length, share_weight=share_weight, bos_id=self.vocab.bos_id, eos_id=self.vocab.eos_id, use_low_rank=use_low_rank
        )
        self.switch_loss = LabelSmoothingLoss(size=4, smoothing=0, padding_idx=0)
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
        switch.masked_fill_((output_token.ge(1) & output_token.le(10)), 3)  # other = 3
        return input_token, output_token, token_length, token_mask, token_self_attention_mask, switch

    def _rebuild_target(self, target, target_length):
        """
        add eos & bos into original token in a batched tensor way
        """
        input_ = t.nn.functional.pad(target, (1, 0), value=self.vocab.bos_id)
        target_ = t.nn.functional.pad(target, (0, 1), value=self.vocab.pad_id)
        indices = t.LongTensor([[i, v.item()] for i, v in enumerate(target_length)]).to(target.device)
        values = t.LongTensor([self.vocab.eos_id for i in target_length]).to(target.device)
        target_ = target_.index_put(tuple(indices.t()), values=values)
        return input_.detach(), target_.detach(), target_length + 1

    def forward(self, feature, feature_length, ori_token, ori_token_length, cal_ce_loss=True):
        #
        t.cuda.empty_cache()
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(
            feature, feature_length, restrict_left_length=self.restrict_left_length,
            restrict_right_length=self.restrict_right_length)
        #
        input_token, output_token, token_length, token_mask, token_self_attention_mask, swich_target = self._prepare_token(
            ori_token, ori_token_length)
        #
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)
        #
        spec_output = self.encoder_linear(spec_feature)
        #
        dot_attention_mask = Masker.get_dot_mask(token_mask, feature_mask)
        output, swich = self.token_decoder(
            input_token, spec_feature, token_mask, token_self_attention_mask, dot_attention_mask)
        switch_loss = self.switch_loss(swich, swich_target)
        if cal_ce_loss:
            ce_loss = self.cal_ce_loss(output, output_token, type='lbce')
        else:
            ce_loss = None
        return output, output_token, spec_output, feature_length, ori_token, ori_token_length, ce_loss, switch_loss

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

    def inference(self, feature, feature_length, decode_type='greedy'):
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(
            feature, feature_length, restrict_left_length=self.restrict_left_length,
            restrict_right_length=self.restrict_right_length
        )
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


    def _encode(self, feature, feature_length):
        feature, feature_mask, feature_self_attention_mask = self._prepare_feature(
            feature, feature_length, restrict_left_length=self.restrict_left_length,
            restrict_right_length=self.restrict_right_length)
        spec_feature = self.spec_encoder(feature, feature_mask, feature_self_attention_mask)
        return spec_feature, feature_mask

    def recognize(self, feature, feature_length, beam=5, penalty=0, ctc_weight=0.3, maxlenratio=0,
                  minlenratio=0, char_list=None, rnnlm=None, lm_weight = 0.1, nbest=1):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) ,length (B)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        assert feature.size(0) == 1
        enc_output, feature_mask = self._encode(feature, feature_length)
        if ctc_weight > 0.0:
            lpz = t.nn.functional.log_softmax(self.encoder_linear(enc_output), -1).squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)
        # print('input lengths: ' + str(h.size(0)))
        # preprare sos
        y = self.vocab.bos_id
        vy = h.new_zeros(1).long()

        if maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(maxlenratio * h.size(0)))
        minlen = int(minlenratio * h.size(0))
        # print('max output length: ' + str(maxlen))
        # print('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:

            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), self.vocab.blank_id, self.vocab.eos_id, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        for i in six.moves.range(maxlen):
            # print('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = Masker.get_mask(t.LongTensor([i+1]))
                ys_self_attention_mask = Masker.get_dot_mask(ys_mask, ys_mask)
                ys_self_attention_mask = Masker.get_forward_mask(ys_self_attention_mask)
                dot_attention_mask = Masker.get_dot_mask(ys_mask, feature_mask)
                ys = t.tensor(hyp['yseq']).unsqueeze(0)
                local_att_scores = self.token_decoder.forward_one_step(ys, enc_output, ys_mask, ys_self_attention_mask, dot_attention_mask)[0]
                local_att_scores = t.nn.functional.log_softmax(local_att_scores, -1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = t.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * t.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = t.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = t.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            # print('number of pruned hypothes: ' + str(len(hyps)))
            # if char_list is not None:
            #     print(
            #         'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                # print('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.vocab.eos_id)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.vocab.eos_id:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            # if end_detect(ended_hyps, i) and maxlenratio == 0.0:
            #     print('end detected at %d', i)
            #     # break

            hyps = remained_hyps
            # if len(hyps) > 0:
            #     # pass
            #     print('remeined hypothes: ' + str(len(hyps)))
            # else:
            #     print('no hypothesis. Finish decoding.')
            #     # break

            # if char_list is not None:
            #     for hyp in hyps:
            #         print(
            #             'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))
            #
            # print('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            # print('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            return None
            # recog_args = Namespace(**vars(recog_args))
            # recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            # return self.recognize(x, recog_args, char_list, rnnlm)

        # print('total log probability: ' + str(nbest_hyps[0]['score']))
        # print('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps