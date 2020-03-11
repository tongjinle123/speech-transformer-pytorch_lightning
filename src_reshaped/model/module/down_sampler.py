from src_reshaped.model.module.down_sample import Conv2dSubsampling
import torch as t


class DownSampler(t.nn.Module):
    def __init__(self, input_size, model_size, dropout):
        super(DownSampler, self).__init__()
        self.input_layer = Conv2dSubsampling(input_size, model_size, dropout=dropout)

    def forward(self, net, input_mask):
        # net b,l,h
        # input_mask b, l
        net, input_mask = self.input_layer(net, input_mask)
        net.masked_fill_(~input_mask.unsqueeze(-1), 0.0)
        return net, input_mask
