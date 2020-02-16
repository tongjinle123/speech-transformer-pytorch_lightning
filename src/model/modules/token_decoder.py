import torch as t
from src.model.modules.transformer_decoder import TransformerDecoder
from src.model.modules.embedding import Embedding


class TokenDecoder(t.nn.Module):
    """
    token transformer decoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, vocab_size,
                 padding_idx, max_length=2048, share_weight=True):
        super(TokenDecoder, self).__init__()

        self.embedding = Embedding(vocab_size, input_size, padding_idx, max_length)
        self.transformer_decoder = TransformerDecoder(input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1/(input_size ** -0.5))
        self.output_layer = t.nn.Linear(input_size, vocab_size, bias=True)
        if share_weight:
            self.output_layer.weight = self.embedding.word_embedding.weight
        else:
            t.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, token_id, encoder_output, token_mask, self_attention_mask, dot_attention_mask):
        net = self.embedding(token_id)
        # net.masked_fill_(token_mask.unsqueeze(-1)==0, 0.0)
        # net *= token_mask.unsqueeze(-1)
        net = self.transformer_decoder(net, token_mask.unsqueeze(-1), encoder_output, self_attention_mask, dot_attention_mask)
        net = self.layer_norm(net)
        net = self.output_layer(net)
        return net
