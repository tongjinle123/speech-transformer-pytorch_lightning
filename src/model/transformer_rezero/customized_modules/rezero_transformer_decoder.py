import torch as t
from src.model.transformer_rezero.customized_modules.rezero_multi_head_attention import MultiHeadAttentionReZeroBlock
from src.model.transformer_rezero.customized_modules.rezero_feed_forward import FeedForwardReZeroBlock


class TransformerDecoder(t.nn.Module):
    """
    transformer decoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank=False):
        super(TransformerDecoder, self).__init__()
        self.layers = t.nn.ModuleList(
            [TransformerDecoderLayer(input_size, feed_forward_size, hidden_size, dropout, num_head, use_low_rank) for _ in range(num_layer)]
        )

    def forward(self, net, src_mask, encoder_output, self_attention_mask, dot_attention_mask):
        no_pad_mask = ~src_mask.unsqueeze(-1)
        for layer in self.layers:
            net = layer(net, no_pad_mask, encoder_output, self_attention_mask, dot_attention_mask)
        return net


class TransformerDecoderLayer(t.nn.Module):
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, use_low_rank=False):
        super(TransformerDecoderLayer, self).__init__()
        self.multi_head_self_attention_block = MultiHeadAttentionReZeroBlock(input_size, hidden_size, dropout, num_head, use_low_rank)
        self.multi_head_dot_attention_block = MultiHeadAttentionReZeroBlock(input_size, hidden_size, dropout, num_head, use_low_rank)
        self.feed_foward_block = FeedForwardReZeroBlock(input_size, feed_forward_size, dropout)

    def forward(self, src, src_mask, encoder_output, self_attention_mask=None, dot_attention_mask=None):
        net = self.multi_head_self_attention_block(src, src, src, self_attention_mask)
        net.masked_fill_(src_mask, 0.0)
        net = self.multi_head_dot_attention_block(net, encoder_output, encoder_output, dot_attention_mask)
        net.masked_fill_(src_mask, 0.0)
        net = self.feed_foward_block(net)
        net.masked_fill_(src_mask, 0.0)
        return net