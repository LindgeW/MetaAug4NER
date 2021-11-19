import torch
from model.mix_seq_tagger import BertSeqTagger
from utils.datautil import load_from, batch_variable, load_data
from utils.dataset import DataLoader
import argparse
import higher

def load_ckpt(ckpt_path, vocab_path, bert_path):
    vocabs = load_from(vocab_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['args_settings']
    model = BertSeqTagger(
        bert_embed_dim=args.bert_embed_dim,
        hidden_size=args.hidden_size,
        num_rnn_layer=1,
        num_tag=len(vocabs['ner']),
        num_bert_layer=args.bert_layer,
        dropout=args.dropout,
        bert_model_path=bert_path
    )
    
    sgd_parameters = [{'params': [p for n, p in model.bert_named_params() if p.requires_grad],
                       'weight_decay': args.weight_decay, 'lr': args.bert_lr},
                        {'params': [p for n, p in model.base_named_params() if p.requires_grad],
                        'weight_decay': args.weight_decay, 'lr': args.learning_rate}]
    meta_opt = torch.optim.SGD(sgd_parameters, lr=args.bert_lr)   
    meta_opt.load_state_dict(ckpt['meta_opt_state'])
    
    model.load_state_dict(ckpt['model_state'])
    model.zero_grad()
    model.eval()
    print('Loading the previous model states ...')
    return model, meta_opt, vocabs


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

    # def write_to_conll(self, wds, golds, preds, split=' '):
    #     assert len(wds) == len(golds) == len(preds)
    #     for wd, gt, pt in zip(wds, golds, preds):
    #         sent = f'{wd}{split}{gt}{split}{pt}\n'
    #         self.fw.write(sent)

    def close(self):
        self.fw.close()


def evaluate(aug_data, val_data, model, vocabs, batch_size=1, save_path='output.txt'):
    aug_loader = DataLoader(aug_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model.eval()
    ner_vocab = vocabs['ner']
    fw = FileWriter(save_path, 'w')
    with torch.no_grad():
        for i, batcher in enumerate(aug_loader):
            batch = batch_variable(batcher, vocabs)
            pred_score = model(batch.bert_inp, batch.mask)
            pred_tag_ids = model.tag_decode(pred_score, batch.mask)
            seq_lens = batch.mask.sum(dim=1).tolist()
            for sid, l in enumerate(seq_lens):
                pred_tags = ner_vocab.idx2inst(pred_tag_ids[sid][1:l].tolist())
                gold_tags = ner_vocab.idx2inst(batch.ner_ids[sid][1:l].tolist())
                fw.write_to_conll(gold_tags, pred_tags)
    fw.close()



def l2rw_test(val_data, aug_data, batch_size, aug_batch_size, model, meta_opt, vocabs):
    train_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    aug_train_loader = DataLoader(aug_data, batch_size=aug_batch_size, shuffle=False)
    train_iter = iter(train_loader)
    for i, batch_train_data in enumerate(aug_train_loader):
        batch = batch_variable(batch_train_data, vocabs)
        
        try:
            batch_val_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_val_data = next(train_iter)
      
        val_batch = batch_variable(batch_val_data, vocabs)

        with torch.backends.cudnn.flags(enabled=False):
            with higher.innerloop_ctx(model, meta_opt) as (meta_model, metaopt):
                yf = meta_model(batch.bert_inp, batch.mask)
                cost = meta_model.tag_loss(yf, batch.ner_ids, batch.mask, reduction='none')
                #eps = torch.zeros(cost.size(), requires_grad=True, device=self.args.device)
                eps = torch.zeros(cost.size(), requires_grad=True)
                meta_train_loss = torch.sum(cost * eps)
                metaopt.step(meta_train_loss)    # differentiable optimizer
                yg = meta_model(val_batch.bert_inp, val_batch.mask)
                meta_val_loss = meta_model.tag_loss(yg, val_batch.ner_ids, val_batch.mask, reduction='mean')
                grad_eps = torch.autograd.grad(meta_val_loss, eps, allow_unused=True)[0].detach()
                del meta_model
                del metaopt

        #w_tilde = torch.clamp(-grad_eps, min=0)
        w_tilde = torch.sigmoid(-grad_eps)
        print(w_tilde)
        w = w_tilde / len(w_tilde)
        #w = w_tilde / (torch.sum(w_tilde) + 1e-8)
        print(w.data.tolist())


# python predictor.py --batch_size 16 --aug_batch_size 32  --val_data_path --aug_data_path --ckpt_path conll_5_model.pkl2 --vocab_path conll_5_vocab.pkl2 --bert_path bert/en_case 

if __name__ == '__main__':
    parse = argparse.ArgumentParser('Inference')
    parse.add_argument('--aug_data_path', type=str, help='augmented data path')
    parse.add_argument('--val_data_path', type=str, help='test data path')
    parse.add_argument('--ckpt_path', type=str, help='checkpoint path')
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--aug_batch_size', type=int, default=16)
    parse.add_argument('--vocab_path', type=str, help='vocab path')
    parse.add_argument('--bert_path', type=str, help='bert path')
    parse.add_argument('--output_path', type=str, default='output.txt', help='output file path')
    args = parse.parse_args()

    model, meta_opt, vocabs = load_ckpt(args.ckpt_path, args.vocab_path, args.bert_path)
    aug_data = load_data(args.aug_data_path)
    val_data = load_data(args.val_data_path)
    #evaluate(aug_data, val_data, model, vocabs, args.batch_size, args.output_path)
    l2rw_test(val_data, aug_data, args.batch_size, args.aug_batch_size, model, meta_opt, vocabs)
    print('Done !!')


