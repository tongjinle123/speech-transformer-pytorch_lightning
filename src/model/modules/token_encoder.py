import torch as t
from src.model.modules.transformer_encoder import TransformerEncoder
from src.model.modules.embedding import Embedding


class TokenEncoder(t.nn.Module):
    """
    token transformer encoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, vocab_size,
                 padding_idx, max_length=2048, share_weight=True):
        super(TokenEncoder, self).__init__()
        self.embedding = Embedding(vocab_size, input_size, padding_idx, max_length, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=(1/(input_size ** -0.5)))
        self.output_layer = t.nn.Linear(input_size, vocab_size, bias=True)
        t.nn.init.xavier_normal_(self.output_layer.weight)
        if share_weight:
            self.output_layer.weight = self.embedding.word_embedding.weight
            self.scale = (input_size ** -0.5)
        else:
            self.scale = 1

    def forward(self, token_id, token_mask, self_attention_mask):
        net = self.embedding(token_id)
        net *= token_mask.unsqueeze(-1)
        net = self.transformer_encoder(net, token_mask.unsqueeze(-1), self_attention_mask)
        net = self.layer_norm(net)
        net = self.output_layer(net)
        return net
