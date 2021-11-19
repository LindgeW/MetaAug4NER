import torch
import torch.nn as nn

'''
LSTMCell
输入: input, (h_0, c_0)
    input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable
    h_0 (batch, hidden_size): 保存着batch中每个元素的初始化隐状态的Tensor
    c_0 (batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出：h_1, c_1
    h_1 (batch, hidden_size): 下一个时刻的隐状态。
    c_1 (batch, hidden_size): 下一个时刻的细胞状态。

GRUCell
输入: input, h0
    input (batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable
    h_0 (batch, hidden_size): 保存着batch中每个元素的初始化隐状态的Tensor
    c_0 (batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出：h_1
    h_1 (batch, hidden_size): 下一个时刻的隐状态。
    c_1 (batch, hidden_size): 下一个时刻的细胞状态。


LSTM
输入: input, (h_0, c_0)
    input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])
    h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
    c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

输出: output, (h_n, c_n)
    output (seq_len, batch, hidden_size * num_directions): 保存RNN最后一层的输出的Tensor。 如果输入是torch.nn.utils.rnn.PackedSequence，那么输出也是torch.nn.utils.rnn.PackedSequence。
    h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的隐状态。
    c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的细胞状态。
    
GRU
输入: input, h_0
    input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False[source])
    h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor

输出: output, h_n
    output (seq_len, batch, hidden_size * num_directions): 保存RNN最后一层的输出的Tensor。 如果输入是torch.nn.utils.rnn.PackedSequence，那么输出也是torch.nn.utils.rnn.PackedSequence。
    h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着RNN最后一个时间步的隐状态。
'''


class HWRNNEncoder(nn.Module):
    def __init__(self, input_size=0,
                 hidden_size=0,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False,
                 dropout=0.0,
                 rnn_type='lstm'):

        super(HWRNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = 0. if num_layers == 1 else dropout
        self.num_directions = 2 if bidirectional else 1
        self._rnn_types = ['RNN', 'LSTM', 'GRU']

        self.rnn_type = rnn_type.upper()
        assert self.rnn_type in self._rnn_types
        # 获取torch.nn对象中相应的的构造函数
        self._rnn_cell = getattr(nn, self.rnn_type+'Cell')  # getattr获取对象的属性或者方法

        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList()

        self.fw_gate_linears, self.bw_gate_linears = nn.ModuleList(), nn.ModuleList()
        self.fw_trans_linears, self.bw_trans_linears = nn.ModuleList(), nn.ModuleList()

        layer_input_size = self.input_size
        for layer_i in range(self.num_layers):  # 纵向延伸
            self.fw_cells.append(self._rnn_cell(input_size=layer_input_size, hidden_size=self.hidden_size))

            # highway connection
            self.fw_gate_linears.append(nn.Linear(in_features=layer_input_size + self.hidden_size,
                                                  out_features=self.hidden_size))

            self.fw_trans_linears.append(nn.Linear(in_features=layer_input_size,
                                                   out_features=self.hidden_size,
                                                   bias=False))

            if self.bidirectional:
                self.bw_cells.append(self._rnn_cell(input_size=layer_input_size, hidden_size=self.hidden_size))

                self.bw_gate_linears.append(nn.Linear(in_features=layer_input_size + self.hidden_size,
                                                      out_features=self.hidden_size))

                self.bw_trans_linears.append(nn.Linear(in_features=layer_input_size,
                                                       out_features=self.hidden_size,
                                                       bias=False))

            layer_input_size = self.num_directions * self.hidden_size

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        if self.rnn_type == 'LSTM':
            hidden = (hidden, hidden)
        return hidden

    def _forward_mask(self, layer_id, cell, inputs, mask, init_hidden):
        out_fw = []
        seq_len = inputs.size(0)  # [seq_len, batch_size, input_size]
        hx_fw = init_hidden
        # print('cell: ', next(cell.parameters()).is_cuda)
        for t in range(seq_len):  # 横向延伸
            if self.rnn_type == 'LSTM':
                h_next, c_next = cell(input=inputs[t], hx=hx_fw)
                # init_hidden[0] -> h0   init_hidden[1] -> c0
                h_next = h_next * mask[t] + init_hidden[0] * (1 - mask[t])
                c_next = c_next * mask[t] + init_hidden[1] * (1 - mask[t])
                # highway
                gate_fw = self.fw_gate_linears[layer_id](torch.cat((inputs[t], hx_fw[0]), dim=-1)).sigmoid()
                h_next = gate_fw * h_next + (1 - gate_fw) * self.fw_trans_linears[layer_id](inputs[t])
                out_fw.append(h_next)
                hx_fw = (h_next, c_next)
            else:
                h_next = cell(input=inputs[t], hx=hx_fw)
                h_next = h_next * mask[t] + init_hidden * (1 - mask[t])
                # highway
                gate_fw = self.fw_gate_linears[layer_id](torch.cat((inputs[t], hx_fw), dim=-1)).sigmoid()
                h_next = gate_fw * h_next + (1 - gate_fw) * self.fw_trans_linears[layer_id](inputs[t])
                out_fw.append(h_next)
                hx_fw = h_next

        out_fw = torch.stack(tuple(out_fw), dim=0)
        return out_fw, hx_fw

    def _backward_mask(self, layer_id, cell, inputs, mask, init_hidden):
        out_bw = []
        seq_len = inputs.size(0)
        hx_bw = init_hidden
        for t in reversed(range(seq_len)):
            if self.rnn_type == 'LSTM':
                h_next, c_next = cell(input=inputs[t], hx=hx_bw)
                h_next = h_next * mask[t] + init_hidden[0] * (1 - mask[t])
                c_next = c_next * mask[t] + init_hidden[1] * (1 - mask[t])
                # highway
                gate_bw = self.bw_gate_linears[layer_id]((torch.cat((inputs[t], hx_bw[0]), dim=-1))).sigmoid()
                h_next = gate_bw * h_next + (1 - gate_bw) * self.bw_trans_linears[layer_id](inputs[t])
                out_bw.append(h_next)
                hx_bw = (h_next, c_next)
            else:
                h_next = cell(input=inputs[t], hx=hx_bw)
                h_next = h_next * mask[t] + init_hidden * (1 - mask[t])
                # highway lstm
                gate_bw = self.bw_gate_linears[layer_id]((torch.cat((inputs[t], hx_bw), dim=-1))).sigmoid()
                h_next = gate_bw * h_next + (1 - gate_bw) * self.bw_trans_linears[layer_id](inputs[t])
                out_bw.append(h_next)
                hx_bw = h_next

        out_bw.reverse()
        out_bw = torch.stack(tuple(out_bw), dim=0)
        return out_bw, hx_bw

    def forward(self, inputs, init_hidden=None, mask=None):
        '''
        :param inputs: (bz, seq_len, input_size)
        :param mask:  (bz, seq_len)
        :param init_hidden:
        :return:
        '''
        if mask is None:  # 不需要掩码，全部都是1
            mask = inputs.new_ones(inputs.shape[:2])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # [batch_size, seq_len] -> [batch_size, seq_len, 1] -> [batch_size, seq_len, hidden_size]
        mask = mask.unsqueeze(2).expand((-1, -1, self.hidden_size))

        batch_size = inputs.size(1)
        if init_hidden is None:
            init_hidden = self.init_hidden(batch_size, device=inputs.device)
            # init_hidden = inputs.data.new(batch_size, self.hidden_size).zero_()
            # if self.rnn_type == 'LSTM':
            # 	init_hidden = (init_hidden, init_hidden)

        hn, cn = [], []
        hx = init_hidden
        for layer in range(self.num_layers):
            # input_drop_mask, hidden_drop_mask = None, None
            seq_len, batch_size, input_size = inputs.shape
            if self.training:
                # drop input
                input_drop_mask = torch.zeros(batch_size, input_size, device=inputs.device).fill_(1 - self.dropout)
                # 在相同的设备上创建一个和inputs数据类型相同的tensor
                # input_drop_mask = inputs.data.new_full((batch_size, input_size), 1 - self.dropout)
                input_drop_mask = torch.bernoulli(input_drop_mask)
                input_drop_mask = torch.div(input_drop_mask, (1 - self.dropout))
                input_drop_mask = input_drop_mask.unsqueeze(-1).expand((-1, -1, seq_len)).permute((2, 0, 1))
                inputs = inputs * input_drop_mask

            out_fw, hidden_fw = self._forward_mask(layer, cell=self.fw_cells[layer],
                                                   inputs=inputs,
                                                   mask=mask,
                                                   init_hidden=hx)

            out_bw, hidden_bw = None, None
            if self.bidirectional:
                out_bw, hidden_bw = self._backward_mask(layer, cell=self.bw_cells[layer],
                                                        inputs=inputs,
                                                        mask=mask,
                                                        init_hidden=hx)
            if self.rnn_type == 'LSTM':
                hn.append(torch.cat((hidden_fw[0], hidden_bw[0]), dim=1) if self.bidirectional else hidden_fw[0])
                cn.append(torch.cat((hidden_fw[1], hidden_bw[1]), dim=1) if self.bidirectional else hidden_fw[1])
            else:
                hn.append(torch.cat((hidden_fw, hidden_bw), dim=1) if self.bidirectional else hidden_fw)

            inputs = torch.cat((out_fw, out_bw), dim=2) if self.bidirectional else out_fw

        if self.rnn_type == 'LSTM':
            hidden = (torch.stack(tuple(hn), dim=0), torch.stack(tuple(cn), dim=0))
        else:
            hidden = torch.stack(tuple(hn), dim=0)

        output = inputs.transpose(0, 1) if self.batch_first else inputs

        return output, hidden


if __name__ == '__main__':
    rnn = HWRNNEncoder(input_size=100, hidden_size=50, num_layers=4, bidirectional=True)
    x = torch.randn(3, 10, 100)
    print(rnn(x)[0])

