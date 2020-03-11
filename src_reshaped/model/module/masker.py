import torch as t
import torch

from distutils.version import LooseVersion

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')
datatype = torch.bool if is_torch_1_2_plus else torch.uint8


class Masker:
    """
    make mask with langth
    """
    def __init__(self):
        pass

    @classmethod
    def get_mask(cls, length):
        mask = make_non_pad_mask(length)
        return mask.to(length.device)

    @classmethod
    def get_dot_mask(cls, query_mask, key_mask):
        return query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)

    @classmethod
    def get_forward_mask(cls, attention_mask):
        attention_mask = t.tril(attention_mask)
        return attention_mask

    @classmethod
    def get_restricted_mask(cls, attention_mask, left=None, right=None):
        if left is None and right is None:
            return attention_mask
        else:
            if left is not None:
                a = t.triu(attention_mask, -left)
            if right is not None:
                b = t.tril(a, right)
        return b

    @classmethod
    def subsequent_mask(cls, size, device="cpu", dtype=datatype):
        """Create mask for subsequent steps (1, size, size).

        :param int size: size of mask
        :param str device: "cpu" or "cuda" or torch.Tensor.device
        :param torch.dtype dtype: result dtype
        :rtype: torch.Tensor
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
        """
        ret = torch.ones(size, size, device=device, dtype=dtype)
        return t.tril(ret, out=ret)


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)


def make_pad_mask(lengths, xs=None, length_dim=-1):

    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None
                    for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


