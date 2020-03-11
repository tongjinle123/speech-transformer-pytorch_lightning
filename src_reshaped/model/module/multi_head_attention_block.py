import torch as t
import torch.nn.functional as F
import numpy as np
from src_reshaped.model.module.low_rank_linear import LowRankLinear


class MultiHeadAttentionBLock(t.nn.Module):
    """
    multi head attention layer combine with layernorm
    """
    def __init__(self, input_size, hidden_size, dropout, num_head, use_low_rank=False):
        super(MultiHeadAttentionBLock, self).__init__()
        if not use_low_rank:
            self.multi_head_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        else:
            self.multi_head_attention = MultiHeadAttentionLowRank(input_size, hidden_size, dropout, num_head)
        self.layer_norm = t.nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask=None):
        residual = query
        query = self.layer_norm(query)
        net = self.multi_head_attention(query, key, value, attention_mask)
        net = self.dropout(net)
        net += residual
        return net


class MultiHeadAttentionLowRank(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head, rank = 128):
        super(MultiHeadAttentionLowRank, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.output_dim = input_size
        self.key_projection = LowRankLinear(input_size, self.num_head * self.hidden_size, rank)
        self.query_projection = LowRankLinear(input_size, self.num_head * self.hidden_size, rank)
        # self.value_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        # t.nn.init.xavier_normal_(self.value_projection.weight)
        self.scale = np.sqrt(self.hidden_size)
        self.linear = LowRankLinear(self.num_head * self.hidden_size, input_size, rank)

    def forward(self, query, key, value, attention_mask=None):
        # key = value
        batch_size, query_lenth, query_dim = query.size()
        key_lenth = key.size(1)
        query_projection = self.query_projection(query).view(batch_size, query_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, QL, H
        key_projection = self.key_projection(key).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, H, KL
        # value_projection = self.value_projection(value).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, KL, H
        attention_matrix = (query_projection @ key_projection) / self.scale
        # B, N, QL, KL
        if attention_mask is not None:
            attention_matrix.masked_fill_(~attention_mask.unsqueeze(1), -float('inf'))

        attention_matrix = F.softmax(attention_matrix, -1)
        attention_matrix = attention_matrix.masked_fill(t.isnan(attention_matrix), 0)
        attention_matrix = self.dropout(attention_matrix)
        weighted = attention_matrix @ key_projection.transpose(-1, -2)
        # B, N, QL, KL * B, N, KL, H -> B, N，QL, H
        output = weighted.permute(0, 2, 1, 3).contiguous().view(batch_size, query_lenth, self.num_head * self.hidden_size)
        output = self.linear(output)
        return output


class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.output_dim = input_size
        self.key_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        self.query_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        # self.value_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        t.nn.init.xavier_normal_(self.key_projection.weight)
        t.nn.init.xavier_normal_(self.query_projection.weight)
        # t.nn.init.xavier_normal_(self.value_projection.weight)
        self.scale = np.sqrt(self.hidden_size)
        self.linear = t.nn.Linear(self.num_head * self.hidden_size, input_size)
        t.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, query, key, value, attention_mask=None):
        # key = value
        batch_size, query_lenth, query_dim = query.size()
        key_lenth = key.size(1)
        query_projection = self.query_projection(query).view(batch_size, query_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, QL, H
        key_projection = self.key_projection(key).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, H, KL
        # value_projection = self.value_projection(value).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, KL, H
        attention_matrix = (query_projection @ key_projection) / self.scale
        # B, N, QL, KL
        if attention_mask is not None:
            attention_matrix.masked_fill_(~attention_mask.unsqueeze(1), -float('inf'))

        attention_matrix = F.softmax(attention_matrix, -1)
        attention_matrix = attention_matrix.masked_fill(t.isnan(attention_matrix), 0)
        attention_matrix = self.dropout(attention_matrix)
        weighted = attention_matrix @ key_projection.transpose(-1, -2)
        # B, N, QL, KL * B, N, KL, H -> B, N，QL, H
        output = weighted.permute(0, 2, 1, 3).contiguous().view(batch_size, query_lenth, self.num_head * self.hidden_size)
        output = self.linear(output)
        return output
