# import torch as t
# from torch.nn.utils.rnn import pad_sequence
#
#
# class Masker:
#     """
#     make mask with langth
#     """
#     def __init__(self):
#         pass
#
#     @classmethod
#     def get_mask(cls, length):
#         mask = pad_sequence([t.ones(i) for i in length.tolist()],  batch_first=True).to(length.device)
#         return mask.detach()
#
#     @classmethod
#     def get_dot_mask(cls, query_mask, key_mask):
#         return query_mask.unsqueeze(-2).bool()
#
#     @classmethod
#     def get_forward_mask(cls, attention_mask):
#         steps = attention_mask.size(-1)
#         seq_mask = t.ones([steps, steps], device=attention_mask.device)
#         seq_mask = t.tril(seq_mask).bool()
#         seq_mask = seq_mask.unsqueeze(0)
#         return seq_mask & attention_mask
#
#     @classmethod
#     def get_restricted_mask(cls, attention_mask, left=None, right=None):
#         assert not (left is None) & (right is None)
#         if left is not None:
#             a = t.triu(attention_mask, -left)
#         if right is not None:
#             b = t.tril(a, right)
#         return b

import torch as t
from torch.nn.utils.rnn import pad_sequence


class Masker:
    """
    make mask with langth
    """
    def __init__(self):
        pass

    @classmethod
    def get_mask(cls, length):
        mask = pad_sequence([t.ones(i) for i in length.tolist()],  batch_first=True).to(length.device)
        return mask.detach()

    @classmethod
    def get_dot_mask(cls, query_mask, key_mask):
        return (query_mask.unsqueeze(-1) @ key_mask.unsqueeze(1))

    @classmethod
    def get_forward_mask(cls, attention_mask):
        attention_mask = t.tril(attention_mask)
        return attention_mask

    @classmethod
    def get_restricted_mask(cls, attention_mask, left=None, right=None):
        assert not (left is None) & (right is None)
        if left is not None:
            a = t.triu(attention_mask, -left)
        if right is not None:
            b = t.tril(a, right)
        return b


