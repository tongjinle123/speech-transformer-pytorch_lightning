import torch as t
from src.model.modules.transformer_encoder import TransformerEncoder
from src.model.modules.vgg_down_sample import Conv2dSubsampling, Conv2dSubsamplingV2, LinearWithPosEmbedding
from src.utils.masker import Masker


class SpecEncoder(t.nn.Module):
    """
    spec encoder
    """
    def __init__(self, input_size, model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, padding_idx, init_size):
        super(SpecEncoder, self).__init__()

        self.input_layer = LinearWithPosEmbedding(input_size, model_size, dropout_rate=0)
        self.layer_norm = t.nn.LayerNorm(model_size, eps=1/(model_size ** -0.5))
        self.transformer_encoder = TransformerEncoder(
            model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer
        )

    def forward(self, net, input_mask, self_attention_mask):
        # net b,l,h
        # input_mask b, l
        net, input_mask = self.input_layer(net, input_mask)
        net = self.transformer_encoder(net, input_mask.unsqueeze(-1), self_attention_mask)
        net = self.layer_norm(net)
        return net

