import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttLayer(nn.Module):
    def __init__(self, d_model, nb_head, dropout=0.1, bias=True):
        super(MultiHeadAttLayer, self).__init__()
        assert d_model % nb_head == 0
        self.d_model = d_model
        self.d_head = d_model // nb_head
        self.nb_head = nb_head
        self.scale = self.d_head ** 0.5
        self.dropout = dropout

        self.fc_q = nn.Linear(d_model, d_model, bias=bias)
        self.fc_k = nn.Linear(d_model, d_model, bias=bias)
        self.fc_v = nn.Linear(d_model, d_model, bias=bias)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.fc_q.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.fc_k.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.fc_v.weight, gain=1 / 2 ** 0.5)
        # nn.init.xavier_uniform_(self.fc_o.weight)

    def forward(self, query, key, value, attn_mask=None):
        '''
        :param query: (bz, q_len, d_model)
        :param key: (bz, k_len, d_model)
        :param value: (bz, v_len, d_model)   len_k == len_v
        :param attn_mask: (bz, 1, 1, k_len)  0 for padding
        :return:
        '''
        bz = query.size(0)
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        # (bz, nb_head, len_q, d_head)
        q_fc = q.reshape(bz, -1, self.nb_head, self.d_head).transpose(1, 2).contiguous()
        # (bz, nb_head, len_k, d_head)
        k_fc = k.reshape(bz, -1, self.nb_head, self.d_head).transpose(1, 2).contiguous()
        # (bz, nb_head, len_v, d_head)
        v_fc = v.reshape(bz, -1, self.nb_head, self.d_head).transpose(1, 2).contiguous()

        # (bz, nb_head, len_q, len_k)
        att_score = torch.matmul(q_fc, k_fc.transpose(-1, -2)) / self.scale

        if attn_mask is not None:
            att_score = att_score.masked_fill(attn_mask == 0, -1e9)

        att_weight = F.softmax(att_score, dim=-1)
        # att_weight = F.dropout(att_weight, p=self.dropout, training=self.training)

        # (bz, nb_head, len_q, d_head)
        att_out = torch.matmul(att_weight, v_fc)

        # (bz, len_q, nb_head * d_head)
        att_out = att_out.transpose(1, 2).contiguous().reshape(bz, -1, self.d_model)

        att_out = F.dropout(att_out, p=self.dropout, training=self.training)

        out = self.layer_norm(att_out + key)

        return out
