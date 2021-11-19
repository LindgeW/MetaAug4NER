import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import sinusoid_encoding, PositionalEmbedding


class MWCNN(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 kernel_sizes=(1, 3, 5),
                 filter_nums=(50, 100, 150),
                 dropout=0.0):
        super(MWCNN, self).__init__()

        self.dropout = dropout

        for k in kernel_sizes:
            assert k % 2 == 1

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=input_size,
                          out_channels=filter_nums[i],
                          kernel_size=k,
                          padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ) for i, k in enumerate(kernel_sizes)
        ])

        self.fc = nn.Linear(sum(filter_nums), output_size)

    def forward(self, x):
        '''
        :param x: (bz, seq_len, dim)
        :return: (bz, dim)
        '''
        conv_in = x.transpose(1, 2).contiguous()
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(conv_in)
            conv_outs.append(conv_out.squeeze(-1))

        output = torch.cat(tuple(conv_outs), dim=-1).contiguous()
        output = F.dropout(output, p=self.dropout, training=self.training)
        return self.fc(output)


class GCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 kernel_size=3, nb_layers=1, dropout=0.0):
        super(GCNN, self).__init__()
        self.dropout = dropout
        self.scale = 0.5 ** 0.5

        # self.pos_embedding = PositionalEmbedding(d_model=input_size)
        # self.pos_embedding = nn.Embedding(3500, input_size, padding_idx=0)
        self.pos_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(sinusoid_encoding(3500, input_size, pad_idx=0)), freeze=True
        )

        self.embed2hn = nn.Linear(input_size, hidden_size)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=hidden_size * 2,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2)
            for _ in range(nb_layers)
        ])

        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        :param x: (bz, seq_len, dim)
        :return: (bz, dim)
        '''

        bs, seq_len, _ = x.size()
        # seq_range = torch.arange(0, seq_len, device=x.device, dtype=x.dtype)
        # pos_embed = self.pos_embedding(seq_range)
        seq_range = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
        pos_embed = self.pos_embedding(seq_range)
        x = x + pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.embed2hn(x)
        # (bz, dim, seq_len)
        conv_in = h.transpose(1, 2).contiguous()
        for conv in self.convs:
            conv_in = F.dropout(conv_in, p=self.dropout, training=self.training)
            conv_out = conv(conv_in)
            conv_out = F.glu(conv_out, dim=1)
            conv_in = (conv_in + conv_out) * self.scale

        # (bs, dim, seq_len) -> (bz, dim, 1) -> (bz, dim)
        conv_out = self.max_pool(conv_in).squeeze(-1)
        # conv_out = F.max_pool1d(conv_in, kernel_size=conv_in.size(-1)).squeeze(-1)
        return self.fc(conv_out)


def max_pool1d(inputs, kernel_size, stride=1, padding='same'):
    '''
    :param inputs: [N, T, C]
    :param kernel_size:
    :param stride:
    :param padding:
    :return: [N, T // stride, C]
    '''
    inputs = inputs.transpose(1, 2).contiguous()  # [N, C, T]
    if padding == 'same':  # 输入输出长度相同
        left = (kernel_size - 1) // 2
        right = (kernel_size - 1) - left
        pad = (left, right)
    else:
        pad = (0, 0)

    inputs = F.pad(inputs, pad)
    outputs = F.max_pool1d(inputs, kernel_size, stride)  # [N, C, T]
    outputs = outputs.transpose(1, 2).contiguous()  # [N, T, C]
    return outputs

























