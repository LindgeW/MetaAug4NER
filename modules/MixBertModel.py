import os
import torch
import torch.nn as nn
from transformers import BertModel
from modules.scale_mix import ScalarMix

# Mixing on the Bert Embedding Layer

class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,

                input_ids2=None,
                attention_mask2=None,
                token_type_ids2=None,
                lambd=None,

                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if input_ids2 is not None and lambd is not None:
            extended_attention_mask2 = self.get_extended_attention_mask(attention_mask2, input_ids2.size(), device)
            len1 = extended_attention_mask.size(-1)
            len2 = extended_attention_mask2.size(-1)
            if len2 > len1:
                tmp_mask = torch.ones_like(extended_attention_mask2) * -10000.
                tmp_mask[:, ..., :len1] = extended_attention_mask
                extended_attention_mask = tmp_mask
            elif len2 < len1:
                tmp_mask = torch.ones_like(extended_attention_mask) * -10000.
                tmp_mask[:, ..., :len2] = extended_attention_mask2
                extended_attention_mask2 = tmp_mask
            extended_attention_mask = torch.max(extended_attention_mask, extended_attention_mask2)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if input_ids2 is not None and lambd is not None:
            embedding_output2 = self.embeddings(
                input_ids=input_ids2,
                position_ids=position_ids,
                token_type_ids=token_type_ids2,
                inputs_embeds=inputs_embeds,
            )

            len1 = embedding_output.size(1)
            len2 = embedding_output2.size(1)
            if len2 > len1:
                tmp_repr = torch.zeros_like(embedding_output2)
                tmp_repr[:, :len1, :] = embedding_output
                embedding_output = tmp_repr
            elif len2 < len1:
                tmp_repr = torch.zeros_like(embedding_output)
                tmp_repr[:, :len2, :] = embedding_output2
                embedding_output2 = tmp_repr
            embedding_output = lambd.unsqueeze(-1) * embedding_output + (1 - lambd).unsqueeze(-1) * embedding_output2

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=1, merge='none', fine_tune=True, use_proj=True, proj_dim=256):
        super(BertEmbedding, self).__init__()
        assert merge in ['none', 'linear', 'mean']
        self.merge = merge
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.fine_tune = fine_tune
        self.bert = MyBertModel.from_pretrained(model_path, output_hidden_states=True)

        self.bert_layers = self.bert.config.num_hidden_layers + 1  # including embedding layer
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size

        if self.merge == 'linear':
            self.scale = ScalarMix(self.nb_layers)
            # self.weighing_params = nn.Parameter(torch.ones(self.num_layers), requires_grad=True)

        if not self.fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        if self.use_proj:
            self.proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
            self.hidden_size = self.proj_dim
        else:
            self.proj = None

    def save_bert(self, save_dir):
        # saved into config file and model
        assert os.path.isdir(save_dir)
        self.bert.save_pretrained(save_dir)
        print('BERT Saved !!!')

    def forward(self, bert_ids, segments, bert_mask, bert_lens,
                bert_ids2=None, segments2=None, bert_mask2=None, bert_lens2=None, lambd=None):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param segments: (bz, bpe_seq_len)  只有一个句子，全0
        :param bert_masks: (bz, bep_seq_len)  经过bpe切词
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :return:
        '''
        all_enc_outs = self.bert(input_ids=bert_ids, token_type_ids=segments, attention_mask=bert_mask,
                                 input_ids2=bert_ids2, token_type_ids2=segments2, attention_mask2=bert_mask2)
        enc_out = all_enc_outs[0]

        seq_mask = bert_lens.gt(0)
        bert_chunks = enc_out[:, :bert_mask.size(1)][bert_mask].split(bert_lens[seq_mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(*bert_lens.shape, self.bert.config.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        output = bert_embed.masked_scatter_(seq_mask.unsqueeze(dim=-1), bert_out)

        if bert_ids2 is not None and lambd is not None:
            seq_mask2 = bert_lens2.gt(0)
            bert_chunks2 = enc_out[:, :bert_mask2.size(1)][bert_mask2].split(bert_lens2[seq_mask2].tolist())
            bert_out2 = torch.stack(tuple([bc.mean(0) for bc in bert_chunks2]))
            bert_embed2 = bert_out2.new_zeros(*bert_lens2.shape, self.bert.config.hidden_size)
            # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
            output2 = bert_embed2.masked_scatter_(seq_mask2.unsqueeze(dim=-1), bert_out2)
            return self.proj(output) if self.proj else output, \
                   self.proj(output2) if self.proj else output2

        return self.proj(output) if self.proj else output
