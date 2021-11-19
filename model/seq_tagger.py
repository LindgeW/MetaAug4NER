import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from modules.BertModel import BertEmbedding
#from modules.BertModel_adapter import BertEmbedding


class BertSeqTagger(nn.Module):
    def __init__(self, bert_embed_dim, hidden_size, num_rnn_layer,
                 num_tag, num_bert_layer=4,
                 dropout=0.0, bert_model_path=None):
        super(BertSeqTagger, self).__init__()
        self.bert_embed_dim = bert_embed_dim
        self.num_tag = num_tag
        self.dropout = dropout
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

        self.tag_crf = CRF(num_tags=num_tag, batch_first=True)

    def bert_params(self):
        return self.bert.bert.parameters()

    def bert_named_params(self):
        return self.bert.bert.named_parameters()

    def base_params(self):
        bert_param_names = []
        for name, param in self.bert.bert.named_parameters():
            if param.requires_grad:
                bert_param_names.append(id(param))

        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append(param)
        return other_params

    def forward(self, bert_inps, mask=None):
        '''
        :param bert_inps: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_embed = self.bert(*bert_inps)
        bert_repr = self.bert_norm(bert_embed)
        if self.training:
            bert_repr = timestep_dropout(bert_repr, p=self.dropout)

        enc_out, hn = self.seq_encoder(bert_repr, non_pad_mask=mask)

        if self.training:
            enc_out = timestep_dropout(enc_out, p=self.dropout)

        tag_score = self.hidden2tag(enc_out)

        return tag_score

        # B, L = enc_out.size()[:2]
        # gate_dist = F.softmax(self.gate_fc(enc_out), dim=-1)
        # # (B, L, 1, C)
        # gate_dist_ = gate_dist.reshape(B*L, 1, self.num_tag)
        # expts = []
        # for efc in self.expt_group:
        #     expts.append(efc(enc_out))
        # # (B, L, C, D)
        # expt_repr = torch.stack(tuple(expts), dim=2).contiguous()
        # expt_repr_ = expt_repr.reshape(B*L, self.num_tag, -1)
        # meta_expt = torch.matmul(gate_dist_, expt_repr_).reshape(B, L, -1)
        # tag_score = self.hidden2tag(meta_expt)
        # return tag_score, gate_dist

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

    def tag_loss(self, tag_score, gold_tags, mask=None, penalty_ws=None, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            lld = self.tag_crf(tag_score, tags=gold_tags, mask=mask, penalty_ws=penalty_ws)
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

