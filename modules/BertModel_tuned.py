from transformers import BertModel
import torch
import os
import torch.nn as nn


class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, head_mask=None, tune_start_layer=12):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if tune_start_layer == 0:
            embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
            last_output, encoder_outputs = self.encoder(embedding_output,
                                                        extended_attention_mask,
                                                        head_mask=head_mask)
            return encoder_outputs
        else:
            with torch.no_grad():  # 第0到tune_start_layer层都不需要梯度，参数不更新
                # self.embeddings.eval()
                embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
                all_hidden_states = (embedding_output,)
                for i in range(tune_start_layer-1):
                    layer_module = self.encoder.layer[i]
                    # layer_module.eval()
                    layer_outputs = layer_module(embedding_output, extended_attention_mask, head_mask[i])

                    embedding_output = layer_outputs[0]
                    all_hidden_states = all_hidden_states + (embedding_output,)

            if tune_start_layer <= self.config.num_hidden_layers:
                for i in range(tune_start_layer-1, self.config.num_hidden_layers):
                    layer_module = self.encoder.layer[i]
                    layer_outputs = layer_module(embedding_output, extended_attention_mask, head_mask[i])

                    embedding_output = layer_outputs[0]
                    all_hidden_states = all_hidden_states + (embedding_output,)

            return all_hidden_states


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers):
        super(BertEmbedding, self).__init__()
        self.bert = MyBertModel.from_pretrained(model_path, output_hidden_states=True)
        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.tune_start_layer = self.bert_layers - nb_layers if nb_layers < self.bert_layers else 0
        self.hidden_size = self.bert.config.hidden_size

        for name, param in self.bert.named_parameters():
            param.requires_grad = False

            items = name.split('.')
            if len(items) < 3:
                continue
            if items[0] == 'embeddings' and 0 >= self.tune_start_layer:
                param.requires_grad = True
            if items[0] == 'encoder' and items[1] == 'layer':
                layer_id = int(items[2]) + 1
                if layer_id >= self.tune_start_layer:
                    param.requires_grad = True

    def save_bert(self, save_dir):
        assert os.path.isdir(save_dir)
        self.bert.save_pretrained(save_dir)
        print('BERT Saved !!!')

    def forward(self, bert_ids, segments, bert_masks, bert_lens):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param segments: (bz, bpe_seq_len)  只有一个句子，全0
        :param bert_masks: (bz, bep_seq_len)  经过bpe切词
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        bert_masks = bert_masks.type_as(mask)

        all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_masks, tune_start_layer=self.tune_start_layer)
        bert_out = all_enc_outs[-1]  # 取倒数第1层

        # 根据bert piece长度切分
        bert_chunks = bert_out[bert_masks].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))  # mask掉CLS
        bert_embed = bert_out.new_zeros(bz, seq_len, self.hidden_size)
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        return bert_embed


