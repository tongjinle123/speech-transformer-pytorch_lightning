import torch as t
from src_reshaped.model.module.gelu import Gelu
import torch
from src_reshaped.model.module.low_rank_linear import LowRankLinear


class FeedForwardBlock(t.nn.Module):
    """
    feed forward layer combine with add - layer norm - dropout
    """
    def __init__(self, input_size, inner_size, dropout, use_low_rank=False):
        super(FeedForwardBlock, self).__init__()
        if not use_low_rank:
            self.linear1 = t.nn.Linear(input_size, inner_size, bias=True)
            t.nn.init.xavier_normal_(self.linear1.weight)
        else:
            self.linear1 = LowRankLinear(input_size, inner_size, rank=128)
        self.gelu = Gelu()
        self.dropout = t.nn.Dropout(dropout, inplace=True)
        if not use_low_rank:
            self.linear2 = t.nn.Linear(inner_size, input_size, bias=True)
            t.nn.init.xavier_normal_(self.linear2.weight)
        else:
            self.linear2 = LowRankLinear(inner_size, input_size, rank=128)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = t.nn.Dropout(dropout, inplace=True)

    def forward(self, net):
        residual = net
        net = self.layer_norm(net)
        net = self.linear1(net)
        net = self.gelu(net)
        net = self.dropout(net)
        net = self.linear2(net)
        net = self.dropout(net)
        net += residual
        return net


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.
    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.
        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).
        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.
    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.
        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size,
                                   stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).
        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))