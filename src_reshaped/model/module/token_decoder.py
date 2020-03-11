import torch as t
from src_reshaped.model.module.transformer_decoder import TransformerDecoder
from src_reshaped.model.module.embedding import Embedding


class TokenDecoder(t.nn.Module):
    """
    token transformer decoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, vocab_size,
                 padding_idx, max_length=50, bos_id=3, eos_id=4, use_low_rank=False, share_weight=True):
        super(TokenDecoder, self).__init__()
        self.max_length = max_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.embedding = Embedding(
            vocab_size, input_size, padding_idx, max_length, scale_word_embedding=share_weight)
        self.transformer_decoder = TransformerDecoder(
            input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1e-6)

    def forward(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
        net = self.embedding(token_id)
        net.masked_fill_(~token_mask.unsqueeze(-1), 0.0)
        net = self.transformer_decoder(net, token_mask, encoder_output, self_attention_mask, dot_attention_mask)
        net = self.layer_norm(net)
        return net

    def forward_one_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
        net = self.embedding(token_id)
        net.masked_fill_(~token_mask.unsqueeze(-1), 0.0)
        net = self.transformer_decoder(net, token_mask, encoder_output, self_attention_mask, dot_attention_mask)
        net = self.layer_norm(net)
        return net

    #
    # def beam_decode_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
    #     net, _ = self.forward(token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
    #     net = t.nn.functional.log_softmax(net, -1)
    #     return net[:, -1, :]
    #
    # def decode_step(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask, topk=1,
    #                 return_last=True):
    #     # token_id B, Lt
    #     # encoder_output B, Lf, H
    #     # token_mask B, Lt
    #     # self_attention_mask B, 1, Lt
    #     # dot_attention_mask B, 1, Lf
    #     net, _ = self.forward(token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask)
    #     net = t.nn.functional.log_softmax(net, -1)
    #     probs, indexs = t.topk(net, topk)
    #     if return_last:
    #         return probs[:, -1, :], indexs[:, -1, :]
    #     else:
    #         return probs, indexs

