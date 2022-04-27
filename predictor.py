import torch
from model.mix_seq_tagger import BertSeqTagger
from utils.datautil import load_from, batch_variable, load_data
from utils.dataset_old import DataLoader
import argparse


def load_ckpt(ckpt_path, vocab_path, bert_path):
    vocabs = load_from(vocab_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['args_settings']
    model = BertSeqTagger(
        bert_embed_dim=args.bert_embed_dim,
        hidden_size=args.hidden_size,
        num_rnn_layer=args.rnn_depth,
        num_tag=len(vocabs['ner']),
        num_bert_layer=args.bert_layer,
        dropout=args.dropout,
        bert_model_path=bert_path
    )
    
    model.load_state_dict(ckpt['model_state'])
    print('Loading the previous model states ...')
    return model, vocabs


class FileWriter(object):
    def __init__(self, path, mode='w'):
        self.path = path
        self.fw = open(self.path, mode, encoding='utf-8')

    def write_to_conll(self, golds, preds, split=' '):
        assert len(golds) == len(preds)
        for gt, pt in zip(golds, preds):
            sent = f'_{split}{gt}{split}{pt}\n'
            self.fw.write(sent)
        self.fw.write('\n')

    def write_to(self, *inps, split=' '):
        for x in zip(*inps):
            sent = split.join(x) + '\n'
            self.fw.write(sent)
        self.fw.write('\n')

    def close(self):
        self.fw.close()


def evaluate(val_data, model, vocabs, batch_size=1, save_path='output.txt'):
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model.eval()
    ner_vocab = vocabs['ner']
    fw = FileWriter(save_path, 'w')
    with torch.no_grad():
        for i, batcher in enumerate(val_loader):
            batch = batch_variable(batcher, vocabs)
            pred_score = model(batch.bert_inp, batch.mask)
            pred_tag_ids = model.tag_decode(pred_score, batch.mask)
            seq_lens = batch.mask.sum(dim=1).tolist()
            for sid, l in enumerate(seq_lens):
                pred_tags = ner_vocab.idx2inst(pred_tag_ids[sid][1:l].tolist())
                gold_tags = ner_vocab.idx2inst(batch.ner_ids[sid][1:l].tolist())
                fw.write_to_conll(gold_tags, pred_tags)
    fw.close()


def infer(test_data, model, vocabs, batch_size=1, save_path='output.txt'):
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model.eval()
    ner_vocab = vocabs['ner']
    fw = FileWriter(save_path, 'w')
    with torch.no_grad():
        for i, batcher in enumerate(test_loader):
            batch = batch_variable(batcher, vocabs)
            pred_score = model(batch.bert_inp, batch.mask)
            pred_tag_ids = model.tag_decode(pred_score, batch.mask)
            seq_lens = batch.mask.sum(dim=1).tolist()
            for sid, l in enumerate(seq_lens):
                seqs = batcher[sid].chars
                pred_tags = ner_vocab.idx2inst(pred_tag_ids[sid][1:l].tolist())
                assert len(seqs) == len(pred_tags)
                fw.write_to(seqs, pred_tags)
    fw.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser('Inference')
    parse.add_argument('--val_data_path', type=str, help='test data path')
    parse.add_argument('--ckpt_path', type=str, help='checkpoint path')
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--vocab_path', type=str, help='vocab path')
    parse.add_argument('--bert_path', type=str, help='bert path')
    parse.add_argument('--output_path', type=str, default='output.txt', help='output file path')
    args = parse.parse_args()

    model, vocabs = load_ckpt(args.ckpt_path, args.vocab_path, args.bert_path)
    val_data = load_data(args.val_data_path)
    evaluate(val_data, model, vocabs, args.batch_size, args.output_path)
    print('Done !!')


