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
                 padding_idx, max_length=50, share_weight=True, bos_id=3, eos_id=4):
        super(TokenDecoder, self).__init__()
        self.max_length = max_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, input_size, padding_idx, max_length, scale_word_embedding=share_weight)
        self.transformer_decoder = TransformerDecoder(input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1/(input_size ** -0.5))
        self.output_layer = t.nn.Linear(input_size, vocab_size, bias=False)
        if share_weight:
            self.output_layer.weight = self.embedding.word_embedding.weight
        else:
            t.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
        net = self.embedding(token_id)
        net.masked_fill_(token_mask.unsqueeze(-1) == 0, 0.0)
        net = self.transformer_decoder(net, token_mask.unsqueeze(-1), encoder_output, self_attention_mask, dot_attention_mask)
        net = self.layer_norm(net)
        net = self.output_layer(net)
        return net

    def beam_search_decode(self, encoder_output, dot_attention_mask, beam_size):
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        feature_length = encoder_output.size(1)

        self.beam_steper = BeamSteper(
            batch_size, beam_size, self.bos_id, self.eos_id, self.vocab_size, device)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            batch_size*beam_size, feature_length, -1)
        dot_attention_mask = dot_attention_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            batch_size * beam_size, 1, feature_length
        )
        with t.no_grad():
            for i in range(self.max_length):
                try:
                    length = self.beam_steper.length_container
                    token_mask = Masker.get_mask(length)
                    self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)

                    token_id = self.beam_steper.get_first_step_token() if i == 0 else self.beam_steper.token_container.view(batch_size * beam_size, -1)
                    last_prob = self.beam_decode_step(
                        token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask,
                        topk=beam_size, return_last=True)
                    self.beam_steper.step(last_prob)
                except:
                    break
        return self.beam_steper.token_container


    def greedy_decode(self, encoder_output, dot_attention_mask):
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

        # B, 1

    def beam_decode_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask, topk=1,
                    return_last=True):
        net = self.forward(token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
        net = t.nn.functional.log_softmax(net, -1)
        return net[:, -1:, :]

    def decode_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask, topk=1,
                    return_last=True):
        # token_id B, Lt
        # encoder_output B, Lf, H
        # token_mask B, Lt
        # self_attention_mask B, 1, Lt
        # dot_attention_mask B, 1, Lf
        net = self.forward(token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
        net = t.nn.functional.log_softmax(net, -1)
        probs, indexs = t.topk(net, topk)
        if return_last:
            return probs[:, -1, :], indexs[:, -1, :]
        else:
            return probs, indexs

