import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.rnn import LSTM
from modules.dropout import timestep_dropout
import numpy as np


class Embeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embed_weight=None, pad_idx=0, unk_idx=None, dropout=0.0, word_dropout=0.0):
        super(Embeddings, self).__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=pad_idx)

        self.dropout = dropout

        if embed_weight is None:
            self.reset_params()
        else:
            # self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)
            self.embeddings.weight.data.copy_(torch.from_numpy(embed_weight))

        if word_dropout > 0:
            assert unk_idx is not None

    def reset_params(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        with torch.no_grad():
            self.embeddings.weight[self.pad_idx].fill_(0)

    @property
    def requires_grad(self):
        return self.embeddings.weight.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.embeddings.weight.requires_grad = value

    @property
    def weight(self):
        return self.embeddings.weight

    def _drop_word(self, words):
        r"""
        按照一定概率随机将word设置为unk_index，这样可以使得unk这个token得到足够的训练,
        且会对网络有一定的regularize的作用。设置该值时，必须同时设置unk_index
        """
        drop_probs = torch.ones_like(words).float() * self.word_dropout
        # drop_probs = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
        drop_mask = torch.bernoulli(drop_probs).eq(1)  # dropout_word越大，越多位置为1
        pad_mask = words.ne(self.pad_idx)
        mask = drop_mask & pad_mask
        words = words.masked_fill(mask, self.unk_idx)
        return words

    def forward(self, x):
        if self.word_dropout > 0 and self.training:
            x = self._drop_word(x)

        embed = self.embeddings(x)

        if self.training:
            embed = timestep_dropout(embed, p=self.dropout)

        return embed


def sinusoid_encoding(nb_pos, dim, pad_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / dim)

    def get_pos_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(dim)]

    sinusoid_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(nb_pos)], dtype=np.float32)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if pad_idx is not None:
        sinusoid_table[pad_idx] = 0.

    return sinusoid_table


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        '''
        :param pos_seq: (seq_len, )
        :return:
        '''
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat(tuple([sinusoid_inp.sin(), sinusoid_inp.cos()]), dim=-1)
        # return pos_emb[:, None, :]  # (seq_len, bz, dim)
        return pos_emb[None, :, :]    # (bz, seq_len, dim)


class ScaleDotProductAttention(nn.Module):
    def __init__(self, k_dim, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.scale = 1. / k_dim ** 0.5
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        '''
        :param q: (bz, q_len, q_dim)
        :param k: (bz, k_len, k_dim)
        :param v: (bz, v_len, v_dim)
        k_len == v_len  v_dim == q_dim
        :param mask: (bz, k_len)  填充部分为0
        :return: (bz, q_len, v_dim)
        '''
        # (bz, q_len, k_len)
        att_score = torch.matmul(q, k.transpose(1, 2)).mul(self.scale)

        if mask is not None:
            att_mask = ~mask[:, None, :]
            att_score = att_score.masked_fill(att_mask, -1e9)

        att_weights = F.softmax(att_score, dim=-1)
        att_out = torch.matmul(att_weights, v)
        return att_out


class DotProductAttention(nn.Module):
    def __init__(self, k_dim):
        super(DotProductAttention, self).__init__()
        self.scale = 1. / k_dim ** 0.5

    def forward(self, hn, enc_out, mask=None):
        '''
        :param hn: query - rnn的末隐层状态 [batch_size, hidden_size]
        :param enc_out: key - rnn的输出  [batch_size, seq_len, hidden_size]
        :param mask: [batch_size, seq_len] 0对应pad
        :return: att_out [batch_size, hidden_size]
        '''
        # type 1
        # (bz, 1, hidden_size) * (bz, hidden_size, n_step) -> (bz, 1, n_step)
        att_score = torch.matmul(hn.unsqueeze(1), enc_out.transpose(1, 2)).squeeze(1)
        att_score.mul_(self.scale)
        if mask is not None:
            att_score = att_score.masked_fill(~mask, -1e9)
        att_weights = F.softmax(att_score, dim=1)  # (bz, n_step)
        # (bz, 1, n_step) * (bz, n_step, hidden_size) -> (bz, 1, hidden_size)
        att_out = torch.matmul(att_weights.unsqueeze(1), enc_out).squeeze(1)

        '''
        # type 2
        # (bz, hidden_size, 1)
        hidden = hn.reshape(hn.size(0), -1, 1)  
        # (bz, n_step, hidden_size) * (bz, hidden_size, 1) -> (bz, n_step)
        att_score = torch.matmul(enc_out, hidden).squeeze(2) 
        att_score.mul_(self.scale)
        if mask is not None:
            att_score = att_score.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(att_score, dim=1)
        # (bz, hidden_sze, n_step) * (bz, n_step, 1) -> (bz, hidden_size, 1)
        att_out = torch.matmul(enc_out.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        '''
        return att_out


class AdditiveAttention(nn.Module):
    def __init__(self, k_size, v_size, hidden_size=None, bias=True):
        super(AdditiveAttention, self).__init__()

        if hidden_size is None:
            hidden_size = v_size

        self.W1 = nn.Linear(k_size, hidden_size, bias=False)
        self.W2 = nn.Linear(v_size, hidden_size, bias=bias)
        self.V = nn.Linear(hidden_size, 1, bias=False)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, q, v, mask=None):
        '''
        :param q: (bz, hidden_size)
        :param v: (bz, n_step, hidden_size)
        :param mask: (bz, n_step)
        :return:
        '''
        # (bz, 1, hidden_size)
        expand_q = q.unsqueeze(1)
        att_score = self.V(torch.tanh(self.W1(expand_q) + self.W2(v)))
        if mask is not None:
            att_score = att_score.masked_fill(~mask.unsqueeze(-1), -1e9)
        # (bz, n_step, 1)
        att_weights = F.softmax(att_score, dim=1)
        # (bz, n_step)
        attn_dist = att_weights.squeeze(dim=-1)
        # (bz, hidden_size)
        att_out = (att_weights * v).sum(dim=1)
        return att_out, attn_dist


class NonlinearMLP(nn.Module):
    def __init__(self, in_feature, out_feature, activation=None, bias=True):
        super(NonlinearMLP, self).__init__()

        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation

        self.bias = bias
        self.linear = nn.Linear(in_features=in_feature,
                                out_features=out_feature,
                                bias=bias)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.orthogonal_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return self.activation(linear_out)


# class Bilinear(nn.Module):
#     def __init__(self, hidden_size, bias=True):
#         """
#         :param hidden_size: 输入的特征维度
#         """
#         super(Bilinear, self).__init__()
#         self.U = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#         self.has_bias = bias
#         if self.has_bias:
#             self.bias = nn.Parameter(torch.zeros(1))
#         else:
#             self.register_parameter("bias", None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.U)
#         if self.has_bias:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, dep, head):
#         """
#         :param head: arc-head tensor [batch, length, hidden]
#         :param dep: arc-dependent tensor [batch, length, hidden]
#         :return output: tensor [batch, length, length]
#         """
#         output = dep.matmul(self.U)
#         output = output.bmm(head.transpose(-1, -2))
#         if self.has_bias:
#             output = output + self.bias
#         return output

class Bilinear(nn.Module):
    def __init__(self, in_dim1, in_dim2, label_dim=1, use_input_bias=False):
        super(Bilinear, self).__init__()
        self.label_dim = label_dim
        self.use_input_bias = use_input_bias

        if self.use_input_bias:
            in_dim1 += 1
            in_dim2 += 1

        self.U = nn.Parameter(torch.randn(label_dim, in_dim1, in_dim2))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x1, x2):
        '''
        :param x1: (bs, len1, in_dim1)
        :param x2: (bs, len2, in_dim2)
        :return: (bs, len1, len2, label_dim)
        '''
        if self.use_input_bias:  # Biaffine
            # (bs, len1, 1)
            bias1 = x1.new_ones(x1.size()[:-1] + (1, ))
            # (bs, len2, 1)
            bias2 = x2.new_ones(x2.size()[:-1] + (1, ))
            # (bs, len1, in_dim1 + 1)
            x1 = torch.cat((x1, bias1), dim=-1)
            # (bs, len2, in_dim2 + 1)
            x2 = torch.cat((x2, bias2), dim=-1)

        # (bs, 1, len1, in_dim1) * (1/label_dim, in_dim1, in_dim2) -> (bs, 1/label_dim, len1, in_dim2)
        tmp = torch.matmul(x1.unsqueeze(1), self.U)
        # (bs, 1/label_dim, len1, in_dim2) * (bs, 1, in_dim2, len2) -> (bs, 1/label_dim, len1, len2)
        out = torch.matmul(tmp, x2.unsqueeze(1).transpose(2, 3).contiguous())
        final = out.squeeze(1) + self.bias
        if self.label_dim > 1:  # (bs, len1, len2, label_dim)
            final = final.permute(0, 2, 3, 1)
        return final


# class Biaffine(nn.Module):
#     def __init__(self, in1_features, in2_features, num_label=1, bias=True):
#         """
#         :param in1_features: 输入的特征1维度
#         :param in2_features: 输入的特征2维度
#         :param num_label: 类别的个数
#         :param bias: 是否使用bias. Default: ``True``
#         """
#         super(Biaffine, self).__init__()
#         self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
#         self.fc = nn.Linear(in1_features + in2_features, num_label, bias=False)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.bilinear.weight)
#         nn.init.xavier_uniform_(self.fc.weight)
#
#     def forward(self, dep, head):
#         """
#         :param dep: [batch, seq_len, hidden] 输入特征1, 即dep
#         :param head: [batch, seq_len, hidden] 输入特征2, 即head
#         :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
#         """
#         output = self.bilinear(head, dep) + self.fc(torch.cat(tuple([head, dep]), dim=-1).contiguous())
#         return output


# self-adaptive attention
class AdaptiveBilinear(nn.Module):
    def __init__(self):
        super(AdaptiveBilinear, self).__init__()

    def forward(self, x1, x2):
        '''
        :param x1: (b, l1, dim1)
        :param x2: (b, l2, dim2)
        :return:
        '''
        assert x1.size(-1) == x2.size(-1)

        # (b, l1, l1)
        x_1 = F.softmax(x1 @ x1.transpose(1, 2), dim=-1)
        # (b, l2, l2)
        x_2 = F.softmax(x2 @ x2.transpose(1, 2), dim=-1)
        # (b, l1, l2)
        x_12 = x1 @ x2.transpose(1, 2)
        # (b, l1, l2)
        x_12 = x_1 @ x_12 @ x_2.transpose(1, 2)
        return x_12


class Biaffine(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in_features = in_features  # mlp_arc_size / mlp_label_size
        self.out_features = out_features  # 1 / rel_size
        self.bias = bias

        # arc: mlp_size
        # label: mlp_size + 1
        self.linear_input_size = in_features + bias[0]
        # arc: mlp_size * 1
        # label: (mlp_size + 1) * rel_size
        self.linear_output_size = out_features * (in_features + bias[1])

        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
            # dim1 += 1
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)
            # dim2 += 1

        # (bz, len1, dim1+1) -> (bz, len1, linear_output_size)
        affine = self.linear(input1)

        # (bz, len1 * self.out_features, dim2)
        affine = affine.reshape(batch_size, len1 * self.out_features, -1)

        # (bz, len1 * out_features, dim2) * (bz, dim2, len2)
        # -> (bz, len1 * out_features, len2) -> (bz, len2, len1 * out_features)
        biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2).contiguous()

        # (bz, len2, len1, out_features)    # out_features: 1 or rel_size
        biaffine = biaffine.reshape((batch_size, len2, len1, -1)).squeeze(-1)

        return biaffine


'''
class CharCNNEmbedding(nn.Module):
    def __init__(self, nb_embeddings,
                 embed_size=100,  # final embedding dim
                 embed_weight=None,
                 kernel_sizes=(1, 3, 5, 7),
                 filter_nums=(50, 100, 150, 200),
                 dropout=0.2):
        super(CharCNNEmbedding, self).__init__()
        self.dropout = dropout
        for ks in kernel_sizes:
            assert ks % 2 == 1, "Odd kernel is supported!"

        self.char_embedding = Embeddings(num_embeddings=nb_embeddings,
                                         embedding_dim=embed_size,
                                         embed_weight=embed_weight,
                                         pad_idx=0,
                                         dropout=dropout)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=embed_size,
                      out_channels=filter_nums[i],
                      padding=ks // 2,
                      kernel_size=ks),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        ) for i, ks in enumerate(kernel_sizes)])

        self.linear = nn.Linear(in_features=sum(filter_nums),
                                out_features=embed_size)

    def forward(self, char_idxs):
        bz, seq_len, max_wd_len = char_idxs.size()
        reshape_char_idxs = char_idxs.reshape(bz*seq_len, -1)
        char_embed = self.char_embedding(reshape_char_idxs)
        # (bz*seq_len, char_embed_size, ch_seq_len)
        char_embed_ = char_embed.transpose(1, 2).contiguous()
        conv_outs = torch.cat(tuple([conv(char_embed_).squeeze(-1) for conv in self.convs]), dim=-1).contiguous()
        # (bz, seq_len, embed_size)
        embed_out = self.linear(conv_outs).reshape(bz, seq_len, -1)
        return embed_out
'''


class CharCNNEmbedding(nn.Module):
    def __init__(self, nb_embeddings,
                 embed_size=100,  # final embedding dim
                 embed_weight=None,
                 kernel_sizes=(1, 3, 5),
                 filter_nums=(50, 100, 150),
                 dropout=0.2):
        super(CharCNNEmbedding, self).__init__()
        self.dropout = dropout
        for ks in kernel_sizes:
            assert ks % 2 == 1, "Odd kernel is supported!"

        self.char_embedding = Embeddings(num_embeddings=nb_embeddings,
                                         embedding_dim=embed_size,
                                         embed_weight=embed_weight,
                                         pad_idx=0,
                                         dropout=dropout)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=embed_size,
                      out_channels=filter_nums[i],
                      padding=ks // 2,
                      kernel_size=ks),
            nn.ReLU(),
        ) for i, ks in enumerate(kernel_sizes)])

        self.linear = nn.Linear(in_features=sum(filter_nums),
                                out_features=embed_size)

    def forward(self, char_idxs):
        '''
        :param char_idxs: (bz, seq_len)
        :return: (bz, seq_len, embed_size)
        '''
        # (bz, seq_len, char_embed_size)
        char_embed = self.char_embedding(char_idxs)
        char_embed_ = char_embed.transpose(1, 2).contiguous()
        conv_outs = torch.cat(tuple([conv(char_embed_).transpose(1, 2) for conv in self.convs]), dim=-1).contiguous()
        # (bz, seq_len, embed_size)
        embed_out = self.linear(conv_outs)
        return embed_out


class CharLSTMEmbedding(nn.Module):
    def __init__(self, nb_embeddings,
                 embed_size=50,
                 hidden_size=50,
                 bidirectional=True,
                 dropout=0.0):
        super(CharLSTMEmbedding, self).__init__()

        self.char_embedding = Embeddings(num_embeddings=nb_embeddings,
                                         embedding_dim=embed_size,
                                         pad_idx=0,
                                         dropout=dropout)

        self.lstm = LSTM(input_size=embed_size,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True,
                         bidirectional=bidirectional)

        lstm_hidden_size = hidden_size * 2 if bidirectional else hidden_size

        self.linear = nn.Linear(in_features=lstm_hidden_size,
                                out_features=embed_size)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, char_idxs):
        '''
        :param char_idxs: (bz, seq_len, max_wd_len)
        :return: (bz, seq_len, embed_size)
        '''
        bz, seq_len, max_wd_len = char_idxs.size()
        reshape_char_idxs = char_idxs.reshape(bz * seq_len, -1)
        char_mask = reshape_char_idxs.gt(0)  # (bz*seq_len, max_wd_len)
        char_embed = self.char_embedding(reshape_char_idxs)
        lstm_out = self.lstm(char_embed, non_pad_mask=char_mask)[0]  # (bz*seq_len, max_wd_len, hidden_size)
        # lstm_out = F.relu(lstm_out)
        # pad部分置成-inf，以免对max_pool造成干扰
        # 如果是avg_pool，pad部分置成0
        mask_lstm_out = lstm_out.masked_fill(char_mask.unsqueeze(-1), -1e9)
        # mask_lstm_out = mask_lstm_out.transpose(1, 2)  # (bz*seq_len, hidden_size, max_wd_len)
        # out = F.max_pool1d(mask_lstm_out, kernel_size=mask_lstm_out.size(-1)).squeeze(dim=-1)
        out, _ = torch.max(mask_lstm_out, dim=1)

        # (bz, seq_len, embed_size)
        embed_out = self.linear(out).reshape(bz, seq_len, -1)
        return embed_out
