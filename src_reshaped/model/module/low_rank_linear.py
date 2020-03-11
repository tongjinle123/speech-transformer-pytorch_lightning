import torch as t
from src_reshaped.model.module.gelu import Gelu


class LowRankLinear(t.nn.Module):
    def __init__(self, input_size, output_size, rank):
        super(LowRankLinear, self).__init__()
        self.input_linear = t.nn.Linear(input_size, rank, bias=False)
        self.output_linear = t.nn.Linear(rank, output_size, bias=False)
        self.gelu = Gelu()
        t.nn.init.xavier_normal_(self.input_linear.weight)
        t.nn.init.xavier_normal_(self.output_linear.weight)

    def forward(self, input_feature):
        net = self.input_linear(input_feature)
        net = self.gelu(net)
        net = self.output_linear(net)
        return net

