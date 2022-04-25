import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from modules.BertModel import BertEmbedding


class BertSeqTagger(nn.Module):
    def __init__(self, bert_embed_dim, hidden_size, num_rnn_layer,
                 num_tag, num_bert_layer=8,
                 dropout=0.5, bert_model_path=None, use_mixup=True):
        super(BertSeqTagger, self).__init__()
        self.bert_embed_dim = bert_embed_dim
        self.num_tag = num_tag
        self.dropout = dropout
        self.use_mixup = use_mixup
        self.bert = BertEmbedding(bert_model_path, 
                                  num_bert_layer,
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)

        hidden_size = self.bert_embed_dim // 2
        self.seq_encoder = LSTM(input_size=self.bert_embed_dim,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout)
        
        self.hidden2tag = nn.Linear(2*hidden_size, num_tag)
        self.tag_crf = CRF(num_tags=num_tag, batch_first=True)

    def bert_params(self):
        return self.bert.bert.parameters()

    def bert_named_params(self):
        return self.bert.bert.named_parameters()

    def base_named_params(self):
        bert_param_names = []
        for name, param in self.bert.bert.named_parameters():
            bert_param_names.append(id(param))

        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_param_names:
                other_params.append((name, param))
        return other_params

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

    def forward(self, bert_inp1, mask1, bert_inp2=None, mask2=None, mix_lmbd=None):
        '''
        :param bert_inp / bert_inp2: bert_ids, segments, bert_masks, bert_lens
        :param mask1 / mask2: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_repr = self.bert(*bert_inp1)
        bert_repr = F.dropout(bert_repr, p=self.dropout, training=self.training)
        if bert_inp2 is not None and mix_lmbd is not None:
            bert_repr2 = self.bert(*bert_inp2)
            bert_repr2 = F.dropout(bert_repr2, p=self.dropout, training=self.training)
            len1 = bert_repr.size(1)
            len2 = bert_repr2.size(1)
            if len2 > len1:
                tmp_repr = torch.zeros_like(bert_repr2)
               # tmp_mask = mask1.new_zeros(bert_repr2.shape[:2])
                tmp_mask = torch.zeros_like(mask2)
                tmp_repr[:, :len1, :] = bert_repr
                tmp_mask[:, :len1] = mask1
                bert_repr = tmp_repr
                mask1 = tmp_mask
            else:
                tmp_repr = torch.zeros_like(bert_repr)
                # tmp_mask = mask2.new_zeros(bert_repr.shape[:2])
                tmp_mask = torch.zeros_like(mask1)
                tmp_repr[:, :len2, :] = bert_repr2
                tmp_mask[:, :len2] = mask2
                bert_repr2 = tmp_repr
                mask2 = tmp_mask
            
            mask = (mask1 + mask2).gt(0)
            bert_repr = bert_repr * mix_lmbd.unsqueeze(1) + bert_repr2 * (1 - mix_lmbd).unsqueeze(1)
        else:
            mask = mask1
        enc_out = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())[0]
        tag_score = self.hidden2tag(enc_out)
        return tag_score

    def tag_loss(self, tag_score, gold_tags, mask=None, mixup_ws=None, alg='crf', reduction='mean'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            lld = self.tag_crf(tag_score, tags=gold_tags, mask=mask, mixup_ws=mixup_ws, reduction=reduction)
            return lld.neg()
        else:
            sum_loss = F.cross_entropy(tag_score.transpose(1, 2), gold_tags, ignore_index=0, reduction='sum')
            return sum_loss / mask.size(0)

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
            return tag_score.data.argmax(dim=-1) 
