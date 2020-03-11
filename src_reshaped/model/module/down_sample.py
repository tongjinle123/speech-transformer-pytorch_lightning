import torch as t
from src_reshaped.model.module.gelu import Gelu
import math


class Conv2dSubsampling(t.nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(Conv2dSubsampling, self).__init__()
        self.conv = t.nn.Sequential(
            t.nn.Conv2d(1, output_size, 3, 2),
            Gelu(),
            t.nn.Conv2d(output_size, output_size, 3, 2),
            Gelu(),
        )
        t.nn.init.kaiming_normal_(self.conv[0].weight)
        t.nn.init.kaiming_normal_(self.conv[2].weight)
        self.output_layer = t.nn.Sequential(
            t.nn.Linear(output_size * (((input_size - 1) // 2 - 1) // 2), output_size),
            PositionalEncoding(output_size, dropout)
        )
        t.nn.init.xavier_normal_(self.output_layer[0].weight)

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.output_layer(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :-2:2][:, :-2:2]


class PositionalEncoding(t.nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.
        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = t.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(t.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = t.zeros(x.size(1), self.d_model)
        position = t.arange(0, x.size(1), dtype=t.float32).unsqueeze(1)
        div_term = t.exp(t.arange(0, self.d_model, 2, dtype=t.float32) *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: t.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.
        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length
        """
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)