import torch as t
import math
import torch
from src.model.modules.vgg_down_sample import ScaledPositionalEncoding, PositionalEncoding


class Embedding(t.nn.Module):
    """
    combine word embedding and position embedding
    """
    def __init__(self, vocab_size=1000, embedding_size=512, padding_idx=0, max_length=2048, dropout=0.1,
                 scale_word_embedding=True):
        super(Embedding, self).__init__()
        self.word_embedding = t.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.position_embedding = ScaledPositionalEncoding(
            d_model=embedding_size, dropout_rate=dropout, max_len=max_length)
        t.nn.init.xavier_normal_(self.word_embedding.weight)

    def forward(self, word_id):
        embedding = self.word_embedding(word_id)
        embedding = self.position_embedding(embedding)
        return embedding





def _pre_hook(state_dict, prefix, local_metadata, strict,
              missing_keys, unexpected_keys, error_msgs):
    """Perform pre-hook in load_state_dict for backward compatibility.
    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
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
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

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