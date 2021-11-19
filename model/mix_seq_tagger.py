import torch.nn as nn
from modules.rnn import LSTM
from modules.dropout import *
from modules.crf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from modules.BertModel import BertEmbedding
#from modules.BertModel_adapter import BertEmbedding
from modules.meta import MetaModule


class AdditiveAttention(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(AdditiveAttention, self).__init__()
        hidden_size = input_size 
        self.attn_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                        nn.ReLU(),
                                        nn.LayerNorm(hidden_size, eps=1e-6),
                                        nn.Linear(hidden_size, 1, bias=False))
        self.output_fc = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, hs, mask=None):
        '''
        :param hs: (bz, n_step, hidden_size)
        :param mask: (bz, n_step)  1为有效部分，0对应pad
        :return:
        '''
        # (bz, n_step, 1)
        att_score = self.attn_layer(hs)
        #if mask is not None:
        #    att_score = att_score.masked_fill(~mask.unsqueeze(-1), -1e9)
        # (bz, n_step, 1)
        att_weights = F.softmax(att_score, dim=1)
        # (bz, hidden_size)
        context = (att_weights * hs).sum(dim=1)
        context = F.dropout(context, p=0.2, training=self.training)
        return self.output_fc(context)


class MultiHeadAttLayer(nn.Module):
    def __init__(self, d_model, output_size, nb_head, dropout=0.1, bias=True):
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
        self.out_fc = nn.Linear(d_model, output_size)
        nn.init.xavier_uniform_(self.out_fc.weight)
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.fc_q.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.fc_k.weight, gain=1 / 2 ** 0.5)
        nn.init.xavier_uniform_(self.fc_v.weight, gain=1 / 2 ** 0.5)
        # nn.init.xavier_uniform_(self.fc_o.weight)

    def forward(self, x, attn_mask=None):
        '''
        :param query: (bz, q_len, d_model)
        :param key: (bz, k_len, d_model)
        :param value: (bz, v_len, d_model)   len_k == len_v
        :param attn_mask: (bz, 1, 1, k_len)  0 for padding
        :return:
        '''
        bz = x.size(0)
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
 
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
       
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
        #out = self.layer_norm(att_out + key)
        #return out
        return self.out_fc(att_out)


class InstanceWeightLayer(nn.Module):
    def __init__(self, input_size, output_size, f_type='linear'):
        super(InstanceWeightLayer, self).__init__()
        self.f_type = f_type
        if f_type == 'pool' or f_type=='linear':
            self.inst_weight_layer = nn.Linear(input_size, output_size)
            nn.init.xavier_uniform_(self.inst_weight_layer.weight)
        #elif f_type == 'linear':
        #    self.inst_weight_layer = nn.Sequential(nn.Linear(input_size, input_size),
        #                                           nn.ReLU(),
        #                                           nn.LayerNorm(input_size, eps=1e-6),
        #                                           nn.Linear(input_size, output_size))
        elif f_type == 'attn':
            self.inst_weight_layer = AdditiveAttention(input_size, output_size)
            #self.inst_weight_layer = MultiHeadAttLayer(input_size, output_size, 2)
            '''
            self.mem_size = input_size
            self.mem_transformation = nn.Linear(in_features=input_size, out_features=self.mem_size)
            self.char_attn_weight = nn.Parameter(torch.zeros(1, self.mem_size), requires_grad=True)
            self.sentence_fc = nn.Linear(in_features=self.mem_size, out_features=output_size)
            nn.init.xavier_normal_(self.char_attn_weight)
            '''
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
            #if mask is not None:
            #    x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            #x = x.transpose(1, 2).contiguous()
            #pool_x = F.avg_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
            expand_mask = mask.float().unsqueeze(-1)
            pool_x = (x * expand_mask).sum(dim=1) / expand_mask.sum(dim=1)
            #inst_w = torch.sigmoid(self.inst_weight_layer(pool_x))
            inst_w = torch.softmax(self.inst_weight_layer(pool_x), dim=0)
        elif self.f_type == 'attn':
            '''
            sentence_f = torch.tanh(self.mem_transformation(x))
            #sentence_f = F.relu(self.mem_transformation(x))
            char_mem = self._soft_attention(sentence_f, self.char_attn_weight, mask)
            char_mem = char_mem.squeeze(1)
            inst_w = torch.sigmoid(self.sentence_fc(char_mem))
            '''
            print('attn weighting ...')
            #inst_w = torch.sigmoid(self.inst_weight_layer(x, mask))
            inst_w = torch.softmax(self.inst_weight_layer(x, mask), dim=0)
            #inst_w = (-torch.log(torch.softmax(self.inst_weight_layer(x, mask), dim=0)) + 1e-8) / len(x)
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

        hidden_size = 768 // 2
        #hidden_size = self.bert_embed_dim // 2
        self.seq_encoder = LSTM(input_size=768,
                                hidden_size=hidden_size,
                                num_layers=num_rnn_layer,
                                dropout=dropout)
        
        #self.hidden2tag = nn.Linear(self.bert_embed_dim, num_tag)
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


    def forward(self, bert_inps1, mask1, bert_inps2=None, mask2=None, mix_lmbd=None):
        '''
        :param bert_inps / bert_inps2: bert_ids, segments, bert_masks, bert_lens
        :param mask1 / mask2: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_repr = self.bert(*bert_inps1)
        #bert_repr = self.bert_norm(bert_embed)
        bert_repr = F.dropout(bert_repr, p=self.dropout, training=self.training)
        if bert_inps2 is not None and mix_lmbd is not None:
            bert_repr2 = self.bert(*bert_inps2)
            bert_repr2 = F.dropout(bert_repr2, p=self.dropout, training=self.training)
            #bert_repr2 = self.bert_norm(bert_embed2)
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
        
        #enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)
       
        tag_score = self.hidden2tag(enc_out)
        #tag_score = self.hidden2tag(bert_repr)
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

    def tag_loss(self, tag_score, gold_tags, mask=None, penalty_ws=None, mixup_ws=None, alg='crf', reduction='mean'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            #tag_crf = self.aug_tag_crf if use_aug_crf else self.tag_crf
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



