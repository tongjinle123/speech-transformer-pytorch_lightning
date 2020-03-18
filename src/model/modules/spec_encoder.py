import torch as t
from src.model.modules.transformer_encoder import TransformerEncoder
from src.model.modules.vgg_down_sample import LinearWithPosEmbedding, PositionalEncoding, LinearWithPosEmbedding2
from src.model.modules.gelu import Gelu

class SpecEncoder(t.nn.Module):
    """
    spec encoder
    """
    def __init__(self, input_size, model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, padding_idx, init_size, use_low_rank=False, input_type='linear', ln_eps=None):
        super(SpecEncoder, self).__init__()
        if input_type == 'linear':
            self.input_layer = LinearWithPosEmbedding(input_size, model_size, dropout_rate=dropout)
        elif input_type == 'conv2d':
            self.input_layer = Conv2dSubsampling(input_size, model_size, dropout_rate=dropout)
        self.layer_norm = t.nn.LayerNorm(model_size)
        self.transformer_encoder = TransformerEncoder(
            model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank=use_low_rank
        )

    def forward(self, net, input_mask, self_attention_mask):
        # net b,l,h
        # input_mask b, l
        net, input_mask = self.input_layer(net, input_mask)
        net.masked_fill_(~input_mask.unsqueeze(-1), 0.0)
        net = self.transformer_encoder(net, input_mask, self_attention_mask)
        net = self.layer_norm(net)
        return net

class SpecEncoder2(t.nn.Module):
    """
    spec encoder
    """
    def __init__(self, input_size, model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, padding_idx, init_size, use_low_rank=False, input_type='linear', ln_eps=None):
        super(SpecEncoder2, self).__init__()
        if input_type == 'linear':
            self.input_layer = LinearWithPosEmbedding2(input_size, model_size, dropout_rate=dropout)
        elif input_type == 'conv2d':
            self.input_layer = Conv2dSubsampling(input_size, model_size, dropout_rate=dropout)
        self.layer_norm = t.nn.LayerNorm(model_size)
        self.transformer_encoder = TransformerEncoder(
            model_size, feed_forward_size, hidden_size, dropout, num_head, num_layer, use_low_rank=use_low_rank
        )

    def forward(self, net, input_mask, self_attention_mask):
        # net b,l,h
        # input_mask b, l
        net, input_mask = self.input_layer(net, input_mask)
        net.masked_fill_(~input_mask.unsqueeze(-1), 0.0)
        net = self.transformer_encoder(net, input_mask, self_attention_mask)
        net = self.layer_norm(net)
        return net


class Conv2dSubsampling(t.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)
    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        super(Conv2dSubsampling, self).__init__()
        self.conv = t.nn.Sequential(
            t.nn.Conv2d(1, odim, 3, 2),
            Gelu(),
            t.nn.Conv2d(odim, odim, 3, 2),
            Gelu()
        )
        self.out = t.nn.Sequential(
            t.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x
        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :-2:2][:, :-2:2]


