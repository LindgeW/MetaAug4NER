from transformers import BertModel
import torch
import math
import torch.nn as nn
from modules.scale_mix import ScalarMix


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class BertAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BertAdapter, self).__init__()
        self.adapter = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     GELU(),
                                     nn.Linear(hidden_dim, input_dim))

        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, hn):
        x = self.adapter(hn)
        return x + hn


class AdapterBertOutput(nn.Module):
    def __init__(self, orig_base, adapter=None):
        # replace BertOutput and BertSelfOutput
        super(AdapterBertOutput, self).__init__()
        self.orig_base = orig_base
        self.adapter = adapter

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.orig_base.dense(hidden_states)
        hidden_states = self.orig_base.dropout(hidden_states)

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)  # add

        hidden_states = self.orig_base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def unfreeze_bert_adapters(bert_model):
    for n, m in bert_model.named_modules():
        if isinstance(m, (BertAdapter, nn.LayerNorm)):
            for pnm, par in m.named_parameters():
                par.requires_grad = True
    return bert_model


def freeze_model(model, freeze=True):
    for n, p in model.named_parameters():
        p.requires_grad = not freeze
    return model


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=12, use_proj=True, proj_dim=256):
        super(BertEmbedding, self).__init__()
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size
        # self.scale = ScalarMix(self.nb_layers)

        if self.use_proj:
            self.proj = nn.Linear(in_features=self.hidden_size, out_features=proj_dim)
            self.hidden_size = self.proj_dim
        else:
            self.proj = None

        # ->>> adapter
        adapter_size = 256
        # not shared in Trans-layer
        # self.adapters = nn.ModuleList([
        #    nn.ModuleDict({'attn': BertAdapter(self.bert.config.hidden_size, self.bert.config.hidden_size // 3),
        #                   'out': BertAdapter(self.bert.config.hidden_size, self.bert.config.hidden_size // 3)})
        #    for _ in range(self.nb_layers)])

        # shared in Trans-layer
        self.adapters = nn.ModuleList([BertAdapter(self.bert.config.hidden_size, adapter_size)
                                       for _ in range(self.nb_layers)])

        for i, bert_layer in enumerate(self.bert.encoder.layer[-self.nb_layers:]):
            bert_layer.output = AdapterBertOutput(bert_layer.output, self.adapters[i])
            bert_layer.attention.output = AdapterBertOutput(bert_layer.attention.output, self.adapters[i])

        self.bert = freeze_model(self.bert)
        self.bert = unfreeze_bert_adapters(self.bert)
        # ->>> adapter

    def forward(self, bert_ids, bert_lens, bert_mask):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :param bert_mask: (bz, bep_seq_len)  经过bpe切词
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        bert_mask = bert_mask.type_as(mask)

        last_enc_out, _, all_enc_outs = self.bert(bert_ids, attention_mask=bert_mask, return_dict=False)
        # bert_out = self.scale(all_enc_outs[-self.nb_layers:])  # (bz, seq_len, 768)
        bert_out = last_enc_out

        # 根据bert piece长度切分
        bert_chunks = bert_out[bert_mask].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(bz, seq_len, self.bert.config.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        output = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)

        if self.proj:
            return self.proj(output)
        else:
            return output


