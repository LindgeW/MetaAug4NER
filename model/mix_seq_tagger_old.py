import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from modules.BertModel import BertEmbedding
# from modules.BertModel_adapter import BertEmbedding


class InstanceWeightLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(InstanceWeightLayer, self).__init__()
        # self.inst_weight_layer = nn.Linear(input_size, output_size)

        self.inst_weight_layer = nn.Sequential(nn.Linear(input_size, input_size),
                                               nn.ReLU(),
                                               nn.LayerNorm(input_size, eps=1e-6),
                                               nn.Linear(input_size, output_size))

    def forward(self, x):
        inst_w = torch.sigmoid(self.inst_weight_layer(x))
        return inst_w


class BertSeqTagger(nn.Module):
    def __init__(self, bert_embed_dim, hidden_size, num_rnn_layer,
                 num_tag, num_bert_layer=4,
                 dropout=0.0, bert_model_path=None, use_aug_crf=False):
        super(BertSeqTagger, self).__init__()
        self.bert_embed_dim = bert_embed_dim
        self.num_tag = num_tag
        self.dropout = dropout
        self.use_aug_crf = use_aug_crf
        self.bert = BertEmbedding(bert_model_path, num_bert_layer,
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=True)

        self.bert_norm = nn.LayerNorm(self.bert_embed_dim, eps=1e-6)

        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout)

        # self.expt_group = nn.ModuleList([nn.Linear(2*hidden_size, hidden_size) for _ in range(num_tag)])
        # self.gate_fc = nn.Linear(2*hidden_size, num_tag)

        self.hidden2tag = nn.Linear(2*hidden_size, num_tag)

        self.inst_weight_layer = InstanceWeightLayer(2*hidden_size, 1)

        self.tag_crf = CRF(num_tags=num_tag, batch_first=True)

        if self.use_aug_crf:
            self.aug_tag_crf = CRF(num_tags=num_tag, batch_first=True)

    def bert_params(self):
        return self.bert.bert.parameters()

    def bert_named_params(self):
        return self.bert.bert.named_parameters()

    def non_bert_params(self):
        bert_param_names = []
        for name, param in self.bert.bert.named_parameters():
            if param.requires_grad:
                bert_param_names.append(id(param))

        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append(param)
        return other_params

    def forward(self, bert_inps1, mask1, bert_inps2=None, mask2=None, mix_lmbd=None):
        '''
        :param bert_inps / bert_inps2: bert_ids, segments, bert_masks, bert_lens
        :param mask1 / mask2: (bs, seq_len)  0 for padding
        :return:
        '''
        mask = mask1
        bert_embed = self.bert(*bert_inps1)
        bert_repr = self.bert_norm(bert_embed)

        if self.training:
            bert_repr = timestep_dropout(bert_repr, p=self.dropout)

        if bert_inps2 is not None and mix_lmbd is not None:
            bert_embed2 = self.bert(*bert_inps2)
            bert_repr2 = self.bert_norm(bert_embed2)
            if self.training:
                bert_repr2 = timestep_dropout(bert_repr2, p=self.dropout)

            len1 = bert_repr.shape[1]
            len2 = bert_repr2.shape[1]
            if len2 > len1:
                tmp_repr = torch.zeros_like(bert_repr2)
                # tmp_mask = mask1.new_zeros(bert_repr2.shape[:2])
                tmp_mask = torch.zeros_like(mask2)
                tmp_repr[:, :len1, :] += bert_repr
                tmp_mask[:, :len1] = mask1
                bert_repr = tmp_repr
                mask1 = tmp_mask
            else:
                tmp_repr = torch.zeros_like(bert_repr)
                # tmp_mask = mask2.new_zeros(bert_repr.shape[:2])
                tmp_mask = torch.zeros_like(mask1)
                tmp_repr[:, :len2, :] += bert_repr2
                tmp_mask[:, :len2] = mask2
                bert_repr2 = tmp_repr
                mask2 = tmp_mask

            mask = (mask1 + mask2).gt(0)
            bert_repr = bert_repr * mix_lmbd.unsqueeze(1) + bert_repr2 * (1 - mix_lmbd).unsqueeze(1)

        enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask)

        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        tag_score = self.hidden2tag(enc_out)

        inst_w = self.inst_weight_layer(hn.data)

        return tag_score, inst_w

    def attn_layer(self, h, mask=None):
        '''
        :param h: (b, l, d)
        :param mask: (b, l)  0 for padding
        :return:
        '''
        # (b, l, d) * (b, d, l) => (b, l, l)
        attn_score = torch.matmul(torch.tanh(h), h.transpose(1, 2))
        if mask is not None:
            attn_score = attn_score.masked_fill(~mask.unsqueeze(1), -1e9)

        attn_prob = F.softmax(attn_score, dim=-1)
        # (b, l, l) * (b, l, d) => (b, l, d)
        attn_out = torch.matmul(attn_prob, h)
        attn_out = F.dropout(attn_out, p=0.2, training=self.training)
        return attn_out

    def tag_loss(self, tag_score, gold_tags, mask=None, penalty_ws=None, use_aug_crf=False, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            if use_aug_crf:
                assert self.use_aug_crf
            tag_crf = self.aug_tag_crf if use_aug_crf else self.tag_crf
            lld = tag_crf(tag_score, tags=gold_tags, mask=mask, penalty_ws=penalty_ws)
            return lld.neg()
        else:
            sum_loss = F.cross_entropy(tag_score.transpose(1, 2), gold_tags, ignore_index=0, reduction='sum')
            return sum_loss / mask.sum()

    def tag_decode(self, tag_score, mask=None, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)  emission probs
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg:
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            best_tag_seq = self.tag_crf.decode(tag_score, mask=mask)
            # return best segment tags
            return pad_sequence(best_tag_seq, batch_first=True, padding_value=0)
        else:
            return tag_score.data.argmax(dim=-1) * mask.long()

