import torch as t
from ..modules.transformer_encoder import TransformerEncoder
from src.model.modules.embedding import Embedding


class TransformerLanguageModel(t.nn.Module):
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, vocab_size,
                 padding_idx, max_length=2048):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = Embedding(vocab_size, input_size, padding_idx, max_length)
        self.transformer_encoder = TransformerEncoder(input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer)

    def forward(self, token_id, token_mask, self_attention_mask):
        net = self.embedding(token_id)
        net = self.transformer_encoder(net, token_mask, self_attention_mask)
        return net
