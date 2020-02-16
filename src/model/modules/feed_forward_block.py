import torch as t
from .gelu import Gelu


class FeedForwardBlock(t.nn.Module):
    """
    feed forward layer combine with add - layer norm - dropout
    """
    def __init__(self, input_size, inner_size, dropout):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = t.nn.Linear(input_size, inner_size, bias=True)
        self.gelu = Gelu()
        self.dropout = t.nn.Dropout(dropout)
        self.linear2 = t.nn.Linear(inner_size, input_size, bias=True)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1/(input_size ** -0.5))
        self.dropout = t.nn.Dropout(dropout)
        t.nn.init.xavier_normal_(self.linear1.weight)
        t.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, net):
        residual = net
        net = self.layer_norm(net)
        net = self.linear1(net)
        net = self.gelu(net)
        # net = t.nn.functional.relu(net)
        net = self.dropout(net)
        net = self.linear2(net)
        net = self.dropout(net)
        net += residual
        return net
