import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from modules.BertModel import BertEmbedding
# from modules.BertModel_adapter import BertEmbedding


class InstanceWeightLayer(nn.Module):
    def __init__(self, input_size, output_size, f_type='linear'):
        super(InstanceWeightLayer, self).__init__()
        self.f_type = f_type
        if f_type == 'pool':
            self.inst_weight_layer = nn.Linear(input_size, output_size)
        elif f_type == 'linear':
            self.inst_weight_layer = nn.Sequential(nn.Linear(input_size, input_size),
                                                   nn.ReLU(),
                                                   nn.LayerNorm(input_size, eps=1e-6),
                                                   nn.Linear(input_size, output_size))
        elif f_type == 'attn':
            self.mem_size = input_size
            self.mem_transformation = nn.Linear(in_features=input_size, out_features=self.mem_size)
            self.char_attn_weight = nn.Parameter(torch.zeros(1, self.mem_size), requires_grad=True)
            self.sentence_fc = nn.Linear(in_features=self.mem_size, out_features=output_size)
            nn.init.xavier_normal_(self.char_attn_weight)

    @staticmethod
    def _soft_attention(text_f, mem_f, mask=None):
        """
        soft attention module
        :param text_f -> torch.FloatTensor, (batch_size, K, dim)
        :param mem_f ->  torch.FloatTensor, (1, dim)
        :param mask -> (batch_size, K)  zero for padding
        :return: mem ->  torch.FloatTensor, (batch, 1, dim)
        """
        att = torch.matmul(text_f, mem_f.transpose(0, 1))   # (bs, K, 1)
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(-1), -1e9)
        weight_mem = F.softmax(att.transpose(1, 2), dim=-1)  # (bs, 1, K)
        mem_align = torch.matmul(weight_mem, text_f)  # (bs, 1, dim)
        return mem_align

    def forward(self, x, mask=None):
        if self.f_type == 'linear':
            inst_w = torch.sigmoid(self.inst_weight_layer(x))
        elif self.f_type == 'pool':
            # (b, l, c) -> (b, c, l)
            x = x.transpose(1, 2).contiguous()
            pool_x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
            inst_w = torch.sigmoid(self.inst_weight_layer(pool_x))
        elif self.f_type == 'attn':
            sentence_f = torch.tanh(self.mem_transformation(x))
            char_mem = self._soft_attention(sentence_f, self.char_attn_weight, mask)
            char_mem = char_mem.squeeze(1)
            return torch.sigmoid(self.sentence_fc(char_mem))
        else:
            inst_w = None

        return inst_w


class BertSeqTagger(nn.Module):
    def __init__(self, bert_embed_dim, hidden_size, num_rnn_layer,
                 num_tag, num_bert_layer=4,
                 dropout=0.0, bert_model_path=None, use_mixup=True):
        super(BertSeqTagger, self).__init__()
        self.bert_embed_dim = bert_embed_dim
        self.num_tag = num_tag
        self.dropout = dropout
        self.use_mixup = use_mixup
        self.bert = BertEmbedding(bert_model_path, num_bert_layer,
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)

        #self.bert_norm = nn.LayerNorm(self.bert_embed_dim, eps=1e-6)

        hidden_size = 768 // 2 
        self.seq_encoder = LSTM(input_size=768,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout)

        self.hidden2tag = nn.Linear(2*hidden_size, num_tag)

        self.tag_crf = CRF(num_tags=num_tag, batch_first=True)            

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


    def base_named_params(self):
        bert_param_names = []
        for name, param in self.bert.bert.named_parameters():
            if param.requires_grad:
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


    def forward(self, bert_inps1, mask1, bert_inps2=None, mask2=None, mix_lmbd=None):
        '''
        :param bert_inps / bert_inps2: bert_ids, segments, bert_masks, bert_lens
        :param mask1 / mask2: (bs, seq_len)  0 for padding
        :return:
        '''
        mask = mask1
        bert_repr = self.bert(*bert_inps1)
        #bert_repr = self.bert_norm(bert_embed)

        if self.training:
            # bert_repr = timestep_dropout(bert_repr, p=self.dropout)
            bert_repr = F.dropout(bert_repr, p=self.dropout, training=self.training)

        enc_out = self.seq_encoder(bert_repr, non_pad_mask=mask.cpu())[0]

        if self.training:
            # enc_out = timestep_dropout(enc_out, p=self.dropout)
            enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)

        if bert_inps2 is not None and mix_lmbd is not None:
            bert_repr2 = self.bert(*bert_inps2)
            #bert_repr2 = self.bert_norm2(bert_embed2)
            if self.training:
                # bert_repr2 = timestep_dropout(bert_repr2, p=self.dropout)
                bert_repr2 = F.dropout(bert_repr2, p=self.dropout, training=self.training)

            enc_out2 = self.seq_encoder(bert_repr2, non_pad_mask=mask2.cpu())[0]

            if self.training:
                # enc_out2 = timestep_dropout(enc_out2, p=self.dropout)
                enc_out2 = F.dropout(enc_out2, p=self.dropout, training=self.training)

            len1 = enc_out.size(1)
            len2 = enc_out2.size(1)
            if len2 > len1:
                tmp_repr = torch.zeros_like(enc_out2)
                tmp_repr[:, :len1, :] += enc_out
                enc_out = tmp_repr
                tmp_mask = torch.zeros_like(mask2)
                tmp_mask[:, :len1] = mask1
                mask1 = tmp_mask
            else:
                tmp_repr = torch.zeros_like(enc_out)
                tmp_repr[:, :len2, :] += enc_out2
                enc_out2 = tmp_repr
                tmp_mask = torch.zeros_like(mask1)
                tmp_mask[:, :len2] = mask2
                mask2 = tmp_mask

            mask = (mask1 + mask2).gt(0)
            enc_out = enc_out * mix_lmbd.unsqueeze(1) + enc_out2 * (1 - mix_lmbd).unsqueeze(1)

        tag_score = self.hidden2tag(enc_out)
        return tag_score


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

    def tag_loss(self, tag_score, gold_tags, mask=None, mixup_ws=None, penalty_ws=None, reduction='mean', alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            lld = self.tag_crf(tag_score, tags=gold_tags, mask=mask, mixup_ws=mixup_ws, penalty_ws=penalty_ws, reduction=reduction)
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


