import torch as t
from src.model.modules.transformer_decoder import TransformerDecoder
from src.model.modules.embedding import Embedding
from src.utils.masker import Masker


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
        token_id = t.full((batch_size, 1), fill_value=self.bos_id, dtype=t.long, device=device)
        length = t.LongTensor([1] * batch_size, device=device)
        #         length = t.full((batch_size), fill_value=1, dtype=t.long, device=device)
        probs = t.Tensor().to(device)
        # count = 0
        with t.no_grad():
            for i in range(self.max_length):
                try:
                    token_mask = Masker.get_mask(length)
                    self_attention_mask = Masker.get_dot_mask(token_mask, token_mask)
                    last_prob, last_token_id = self.decode_step(
                        token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask,
                        topk=beam_size, return_last=True)



                except:
                    break
        return None


    def greedy_decode(self, encoder_output, dot_attention_mask):
        """
        batched greedy decode
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        token_id = t.full((batch_size, 1), fill_value=self.bos_id, dtype=t.long, device=device)
        length = t.LongTensor([1] * batch_size, device=device)
        #         length = t.full((batch_size), fill_value=1, dtype=t.long, device=device)
        probs = t.Tensor().to(device)
        # count = 0
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
                    probs = t.cat([probs, last_prob], dim=1)
                    for index, id in enumerate(last_token_id.squeeze(1)):
                        if id != self.eos_id:
                            length[index] += 1
                except:
                    #TODO: to be more consious
                    break

                # print('length',length)
                # count += 1
                # if count ==4:
                #     break
        return token_id

        # B, 1

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

