import torch as t
from .multi_head_attention_block import MultiHeadAttentionBLock
from .feed_forward_block import FeedForwardBlock


class TransformerEncoder(t.nn.Module):
    """
    transformer encoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank=False):
        super(TransformerEncoder, self).__init__()
        self.layers = t.nn.ModuleList(
            [TransformerEncoderLayer(input_size, feed_forward_size, hidden_size, dropout, num_head, use_low_rank) for _ in range(num_layer)]
        )

    def forward(self, net, src_mask, self_attention_mask):
        for layer in self.layers:
            net = layer(net, src_mask, self_attention_mask)
        return net


class TransformerEncoderLayer(t.nn.Module):
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, use_low_rank=False):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention_block = MultiHeadAttentionBLock(input_size, hidden_size, dropout, num_head, use_low_rank)
        self.feed_foward_block = FeedForwardBlock(input_size, feed_forward_size, dropout)

    def forward(self, src, src_mask, self_attention_mask=None):

        net = self.multi_head_attention_block(src, src, src, self_attention_mask)
        net.masked_fill_(src_mask == 0, 0.0)
        net = self.feed_foward_block(net)
        net.masked_fill_(src_mask == 0, 0.0)
        return net