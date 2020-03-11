import torch as t
from src_reshaped.model.module.transformer_encoder import TransformerEncoder
from src_reshaped.model.module.down_sample import Conv2dSubsampling


class SpecEncoder(t.nn.Module):
    """
    spec encoder
    """
    def __init__(self, model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank=False):
        super(SpecEncoder, self).__init__()

        # self.input_layer = Conv2dSubsampling(input_size, model_size, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(
            model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank=use_low_rank
        )
        self.layer_norm = t.nn.LayerNorm(model_size, eps=1e-6)


    def forward(self, net, input_mask, self_attention_mask):
        net = self.transformer_encoder(net, input_mask, self_attention_mask)
        net = self.layer_norm(net)
        return net, input_mask

