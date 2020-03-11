import torch as t
from src.model.modules.spec_augment_layer import SpecAugment
from src.utils.vocab import Vocab
from argparse import Namespace
from distutils.util import strtobool

import logging
import math
import torch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer


class Transformer(t.nn.Module):
    def __init__(self, num_time_mask=2, num_freq_mask=2, freq_mask_length=15, time_mask_length=15, feature_dim=320,
                 model_size=512, feed_forward_size=1024, hidden_size=64, dropout=0.1, num_head=8, num_encoder_layer=6,
                 num_decoder_layer=6, vocab_path='testing_vocab.model', max_feature_length=1024, max_token_length=50,
                 enable_spec_augment=True, share_weight=True, smoothing=0.1, restrict_left_length=20,
                 restrict_right_length=20, mtlalpha=0.2, report_wer=True):
        super(Transformer, self).__init__()

        self.enable_spec_augment = enable_spec_augment
        self.max_token_length = max_token_length
        self.restrict_left_length = restrict_left_length
        self.restrict_right_length = restrict_right_length
        self.vocab = Vocab(vocab_path)
        self.sos = self.vocab.bos_id
        self.eos = self.vocab.eos_id
        self.adim = model_size
        self.odim = self.vocab.vocab_size
        self.ignore_id = self.vocab.pad_id

        if enable_spec_augment:
            self.spec_augment = SpecAugment(
                num_time_mask=num_time_mask, num_freq_mask=num_freq_mask,
                freq_mask_length=freq_mask_length, time_mask_length=time_mask_length,
                max_sequence_length=max_feature_length)

        self.encoder = Encoder(
            idim=feature_dim, attention_dim=model_size, attention_heads=num_head, linear_units=feed_forward_size,
            num_blocks=num_encoder_layer, dropout_rate=dropout, positional_dropout_rate=dropout,
            attention_dropout_rate=dropout, input_layer='linear', padding_idx=self.vocab.pad_id)

        self.decoder = Decoder(
            odim=self.vocab.vocab_size, attention_dim=model_size, attention_heads=num_head,
            linear_units=feed_forward_size, num_blocks=num_decoder_layer, dropout_rate=dropout,
            positional_dropout_rate=dropout, self_attention_dropout_rate=dropout, src_attention_dropout_rate=0,
            input_layer='embed', use_output_layer=False)
        self.decoder_linear = t.nn.Linear(model_size, self.vocab.vocab_size, bias=True)
        self.decoder_switch_linear = t.nn.Linear(model_size, 4, bias=True)

        self.criterion = LabelSmoothingLoss(
            size=self.odim, smoothing=smoothing, padding_idx=self.vocab.pad_id, normalize_length=True)
        self.switch_criterion = LabelSmoothingLoss(
            size=4, smoothing=0, padding_idx=self.vocab.pad_id, normalize_length=True
        )
        self.mtlalpha = mtlalpha
        if mtlalpha > 0.0:
            self.ctc = CTC(self.odim, eprojs=self.adim, dropout_rate=dropout, ctc_type='builtin', reduce=False)
        else:
            self.ctc = None

        if report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculator
            def load_token_list(path=vocab_path.replace('.model', '.vocab')):
                with open(path) as reader:
                    data = reader.readlines()
                    data = [i.split('\t')[0] for i in data]
                return data
            self.char_list = load_token_list()
            self.error_calculator = ErrorCalculator(
                char_list=self.char_list, sym_space=' ', sym_blank=self.vocab.blank_token, report_wer=True)
        else:
            self.error_calculator = None
        self.rnnlm = None
        self.reporter = Reporter()

        self.switch_loss = LabelSmoothingLoss(size=4, smoothing=0, padding_idx=0)
        print('initing')
        initialize(self, init_type='xavier_normal')
        print('inited')

    def build_sample_data(self, feature_dim=320, cuda=False):
        feature = t.randn((2, 120, feature_dim))
        feature_length = t.LongTensor([i for i in range(119, 121)])
        target = t.LongTensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 0]])
        target_length = t.LongTensor([7, 6])
        if cuda:
            return feature.cuda(), feature_length.cuda(), target.cuda(), target_length.cuda()
        else:
            return feature, feature_length, target, target_length

    def get_switch_target(self, ys_out_pad):
        switch = t.ones_like(ys_out_pad, device=ys_out_pad.device).long()  # eng = 1
        switch.masked_fill_(ys_out_pad.eq(0), 0)  # pad=0
        switch.masked_fill_((ys_out_pad.ge(12) & ys_out_pad.le(4211)), 2)  # ch = 2
        switch.masked_fill_((ys_out_pad.ge(1) & ys_out_pad.le(10)), 3)  # other = 3
        return switch

    def forward(self, xs_pad, ilens, ys_pad, ys_length):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        if self.spec_augment is not None:
            if self.training:
                xs_pad = self.spec_augment(xs_pad, ilens)

        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        self.hs_pad = hs_pad
        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, ys_length, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        switch_pred_pad = self.decoder_switch_linear(pred_pad)
        pred_pad = t.nn.functional.log_softmax(self.decoder_linear(pred_pad), -1)

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        ys_switch_pad = self.get_switch_target(ys_out_pad)
        loss_switch = self.switch_criterion(switch_pred_pad, ys_switch_pad)
        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                               ignore_label=self.ignore_id)
        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
                self.cer_ctc = float(cer_ctc)
        # 5. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - 0.1 - alpha) * loss_att + loss_switch * 0.1
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)
            self.loss_att = loss_att_data
            self.loss_ctc = loss_ctc_data
            self.loss_switch = float(loss_switch)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            pass
            # self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        print('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        print('max output length: ' + str(maxlen))
        print('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            import numpy

            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            print('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]
                    local_att_scores = t.nn.functional.log_softmax(self.decoder_linear(local_att_scores), -1)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

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
            print('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                print(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                print('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                print('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                print('remeined hypothes: ' + str(len(hyps)))
            else:
                print('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    print(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            print('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            print('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        print('total log probability: ' + str(nbest_hyps[0]['score']))
        print('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret
