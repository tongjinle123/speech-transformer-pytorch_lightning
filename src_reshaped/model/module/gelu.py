import torch as t
import math


class Gelu(t.nn.Module):
    """
    GELU activation layer
    """
    def __init__(self):
        super(Gelu, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + t.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * t.pow(x, 3))))

