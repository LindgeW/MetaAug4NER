import os
import time
import torch
import random
import numpy as np
# from model.seq_tagger import BertSeqTagger
from model.mix_seq_tagger import BertSeqTagger
from config.conf import args_config, data_config
from utils.dataset import DataLoader, BucketDataLoader, BatchWrapper
from utils.datautil import load_data, create_vocab, batch_variable, save_to
from utils.conlleval import evaluate
import torch.nn.utils as nn_utils
from logger.logger import logger
import higher


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        self.data_config = data_config
        genre = args.genre
        self.train_set = load_data(data_config[genre]['train'])
        self.val_set = load_data(data_config[genre]['dev'])
        self.test_set = load_data(data_config[genre]['test'])
        print('train data size:', len(self.train_set))
        print('validate data size:', len(self.val_set))
        print('test data size:', len(self.test_set))

        if self.args.train_type != 'vanilla':
            self.aug_train_set = load_data(data_config[args.aug_genre]['train'])
            print('aug train data size:', len(self.aug_train_set))
        else:
            self.aug_train_set = None

        self.dev_loader = DataLoader(self.val_set, batch_size=self.args.test_batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size)
        self.vocabs = create_vocab(self.train_set, data_config['pretrained']['bert_model'], embed_file=None)
        save_to(args.vocab_chkp, self.vocabs)

        self.model = BertSeqTagger(
            bert_embed_dim=args.bert_embed_dim,
            hidden_size=args.hidden_size,
            num_rnn_layer=args.rnn_depth,
            num_tag=len(self.vocabs['ner']),
            num_bert_layer=args.bert_layer,
            dropout=args.dropout,
            bert_model_path=data_config['pretrained']['bert_model']
        ).to(args.device)

        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %dM trainable parameters..." % (total_params / 1e6))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.model.bert_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},

            {'params': [p for n, p in self.model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
            {'params': [p for n, p in self.model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.learning_rate}
            # {'params': [p for n, p in self.model.base_named_params()],
            #  'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate}
        ]

        sgd_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()],
             'weight_decay': self.args.weight_decay, 'lr': self.args.bert_lr},

            {'params': [p for n, p in self.model.base_named_params()],
             'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_bert_parameters, lr=self.args.bert_lr, eps=self.args.eps)
        self.meta_opt = torch.optim.SGD(sgd_parameters, lr=self.args.learning_rate)
        # self.meta_opt = torch.optim.SGD(sgd_parameters, lr=self.args.bert_lr, momentum=0.9)
        # self.meta_opt = torch.optim.SGD(optimizer_bert_parameters, lr=self.args.bert_lr, momentum=0.9)

    def train_epoch(self, ep=0):
        print('vanilla training ...')
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        if self.aug_train_set is None:
            train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(self.aug_train_set + self.train_set, batch_size=self.args.batch_size,
                                      shuffle=True)
        self.model.zero_grad()
        for i, batch_train_data in enumerate(train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            tag_score = self.model(batch.bert_inp, batch.mask)
            loss = self.model.tag_loss(tag_score, batch.ner_ids, mask=batch.mask)
            loss.backward()
            loss_val = loss.data.item()
            train_loss += loss_val
            # nn_utils.clip_grad_norm_(self.model.base_params(), max_norm=self.args.grad_clip)
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()), max_norm=self.args.bert_grad_clip)
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=self.args.grad_clip)
            self.optimizer.step()
            self.model.zero_grad()
            logger.info('[Epoch %d] Iter%d time cost: %.2fs, train loss: %.3f' % (
                ep, i, (time.time() - t1), loss_val))
        return train_loss

    def train_mixup(self, ep=0):
        print('mixup training ...')
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_loader = BatchWrapper(
            BucketDataLoader(self.aug_train_set + self.train_set, batch_size=self.args.batch_size,
                             key=lambda x: len(x.chars), shuffle=True, sort_within_batch=True), mixup=True)
        self.model.zero_grad()
        for i, batch_train_data in enumerate(train_loader):
            batcher, mix_alpha, batcher2, _ = batch_train_data
            batch = batch_variable(batcher, self.vocabs)
            batch2 = batch_variable(batcher2, self.vocabs)
            batch.to_device(self.args.device)
            batch2.to_device(self.args.device)
            mix_alpha = mix_alpha.to(self.args.device)

            tag_score = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_alpha)
            loss = self.model.tag_loss(tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                       mixup_ws=mix_alpha)
            loss += self.model.tag_loss(tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                        mixup_ws=1 - mix_alpha)
            loss.backward()
            loss_val = loss.data.item()
            train_loss += loss_val

            # nn_utils.clip_grad_norm_(self.model.base_params(), max_norm=self.args.grad_clip)
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()), max_norm=self.args.bert_grad_clip)
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=self.args.grad_clip)
            self.optimizer.step()
            self.model.zero_grad()
            logger.info('[Epoch %d] Iter%d time cost: %.2fs, train loss: %.3f' % (
                ep, i, (time.time() - t1), loss_val))
        return train_loss

    def train_reweight_epoch(self, ep=0):
        print('train reweighting ...')
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        aug_train_loader = DataLoader(self.aug_train_set + self.train_set, batch_size=self.args.aug_batch_size,
                                      shuffle=True)
        train_iter = iter(train_loader)
        for i, batch_train_data in enumerate(aug_train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            try:
                batch_val_data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_val_data = next(train_iter)
            val_batch = batch_variable(batch_val_data, self.vocabs)
            val_batch.to_device(self.args.device)

            self.meta_opt.zero_grad()
            self.model.zero_grad()
            with torch.backends.cudnn.flags(enabled=False):
                with higher.innerloop_ctx(self.model, self.meta_opt) as (meta_model, meta_opt):
                    # with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                    yf = meta_model(batch.bert_inp, batch.mask)
                    cost = meta_model.tag_loss(yf, batch.ner_ids, batch.mask, reduction='none')
                    eps = torch.zeros(cost.size(), requires_grad=True, device=self.args.device)
                    # eps = 1e-8 * torch.ones(cost.size(), device=self.args.device)
                    # eps = torch.ones(cost.size(), device=self.args.device) / cost.size(0)
                    # eps.requires_grad = True
                    meta_train_loss = torch.sum(cost * eps)
                    meta_opt.step(meta_train_loss)  # differentiable optimizer

                    yg = meta_model(val_batch.bert_inp, val_batch.mask)
                    meta_val_loss = meta_model.tag_loss(yg, val_batch.ner_ids, val_batch.mask, reduction='mean')
                    grad_eps = torch.autograd.grad(meta_val_loss, eps, allow_unused=True)[0].detach()
                    del meta_opt
                    del meta_model

            # w_tilde = torch.clamp(-grad_eps, min=0)
            w_tilde = torch.sigmoid(-grad_eps)
            norm_w = torch.sum(w_tilde)
            if norm_w != 0:
                w = w_tilde / norm_w
            else:
                w = w_tilde
            # w = w_tilde / len(w_tilde)
            yf = self.model(batch.bert_inp, batch.mask)
            cost = self.model.tag_loss(yf, batch.ner_ids, batch.mask, reduction='none')
            batch_loss = torch.sum(cost * w)
            self.model.zero_grad()
            batch_loss.backward()
            loss_val = batch_loss.data.item()
            train_loss += loss_val
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.base_params()),
            #                          max_norm=self.args.grad_clip)
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
            #                          max_norm=self.args.bert_grad_clip)
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=self.args.grad_clip)
            self.optimizer.step()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, train_loss: %.4f' % (
                ep, i, (time.time() - t1), loss_val))
        return train_loss / len(train_loader)

    def train_reweight_mixup(self, ep=0):
        print('train reweighting mixup .....')
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_loader = BatchWrapper(
            BucketDataLoader(self.train_set, batch_size=self.args.batch_size, key=lambda x: len(x.chars), shuffle=True,
                             sort_within_batch=True), mixup=True, mixup_args=(self.args.mix_alpha, self.args.mix_alpha))
        aug_train_loader = BatchWrapper(
            BucketDataLoader(self.aug_train_set + self.train_set, batch_size=self.args.aug_batch_size,
                             key=lambda x: len(x.chars), shuffle=True, sort_within_batch=True), mixup=True,
            mixup_args=(self.args.mix_alpha, self.args.mix_alpha))
        val_iter = iter(train_loader)
        self.model.zero_grad()
        for i, batch_train_data in enumerate(aug_train_loader):
            batcher, mix_alpha, batcher2, _ = batch_train_data
            batch = batch_variable(batcher, self.vocabs)
            batch2 = batch_variable(batcher2, self.vocabs)
            batch.to_device(self.args.device)
            batch2.to_device(self.args.device)
            mix_alpha = mix_alpha.to(self.args.device)
            try:
                batch_val_data = next(val_iter)
            except StopIteration:
                val_iter = iter(train_loader)
                batch_val_data = next(val_iter)

            val_batcher, val_mix_alpha, val_batcher2, _ = batch_val_data
            val_batch = batch_variable(val_batcher, self.vocabs)
            val_batch2 = batch_variable(val_batcher2, self.vocabs)
            val_batch.to_device(self.args.device)
            val_batch2.to_device(self.args.device)
            val_mix_alpha = val_mix_alpha.to(self.args.device)

            self.model.zero_grad()
            self.meta_opt.zero_grad()
            with torch.backends.cudnn.flags(enabled=False):
                with higher.innerloop_ctx(self.model, self.meta_opt) as (meta_model, meta_opt):
                    yf = meta_model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_alpha)
                    cost = meta_model.tag_loss(yf[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                               mixup_ws=mix_alpha, reduction='none')
                    cost += meta_model.tag_loss(yf[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                                mixup_ws=1 - mix_alpha, reduction='none')

                    eps = torch.zeros(cost.size(), requires_grad=True).to(self.args.device)
                    # eps = torch.tensor(1e-8 * torch.ones(cost.size()), requires_grad=True, device=self.args.device)
                    # eps = 1e-8 * torch.ones(cost.size(), device=self.args.device)
                    # eps.requires_grad = True
                    meta_train_loss = torch.sum(cost * eps)
                    meta_opt.step(meta_train_loss)

                    yg = meta_model(val_batch.bert_inp, val_batch.mask, val_batch2.bert_inp, val_batch2.mask,
                                    val_mix_alpha)
                    meta_val_loss = meta_model.tag_loss(yg[:, :val_batch.ner_ids.shape[1]], val_batch.ner_ids,
                                                        val_batch.mask,
                                                        mixup_ws=val_mix_alpha, reduction='mean')
                    meta_val_loss += meta_model.tag_loss(yg[:, :val_batch2.ner_ids.shape[1]], val_batch2.ner_ids,
                                                         val_batch2.mask,
                                                         mixup_ws=1 - val_mix_alpha, reduction='mean')
                    grad_eps = torch.autograd.grad(meta_val_loss, eps, allow_unused=True)[0].detach()
                    del meta_opt
                    del meta_model

            # w_tilde = torch.clamp(-grad_eps, min=0)
            w_tilde = torch.sigmoid(-grad_eps)
            norm_w = torch.sum(w_tilde)
            if norm_w != 0:
                w = w_tilde / norm_w
            else:
                w = w_tilde
            # w = w_tilde / len(w_tilde)

            tag_score = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_alpha)
            cost = self.model.tag_loss(tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                       mixup_ws=mix_alpha, reduction='none')
            cost += self.model.tag_loss(tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                        mixup_ws=1 - mix_alpha, reduction='none')
            batch_loss = torch.sum(cost * w)
            self.model.zero_grad()
            batch_loss.backward()

            val_loss = batch_loss.data.item()
            train_loss += val_loss
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.base_params()),
            #                          max_norm=self.args.grad_clip)
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
            #                          max_norm=self.args.bert_grad_clip)
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=self.args.grad_clip)
            self.optimizer.step()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, train loss: %.4f' % (
                ep, i, (time.time() - t1), val_loss))
        return train_loss / len(aug_train_loader)

    def save_states(self, save_path, best_test_metric=None):
        # if os.path.exists(save_path):
        #    os.remove(save_path)
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.meta_opt.zero_grad()
        check_point = {'best_prf': best_test_metric,
                       'model_state': self.model.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'meta_opt_state': self.meta_opt.state_dict(),
                       'args_settings': self.args}
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')

    def restore_states(self, load_path):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.meta_opt.zero_grad()
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.meta_opt.load_state_dict(ckpt['meta_opt_state'])
        self.args = ckpt['args_settings']
        logger.info('Loading the previous model states ...')
        print('Previous best prf result is: %s' % ckpt['best_prf'])

    def run(self):
        patient = 0
        to_mix, to_rw = False, False
        two_stage = False
        best_dev_metric, best_test_metric = dict(), dict()
        for ep in range(1, 1 + self.args.epoch):
            if not to_rw:
                if not to_mix:
                    # to_mix, to_rw = False, False
                    train_loss = self.train_epoch(ep)
                else:
                    # to_mix, to_rw = True, False
                    train_loss = self.train_mixup(ep)
            else:
                if not to_mix:
                    # to_mix, to_rw = False, True
                    train_loss = self.train_reweight_epoch(ep)
                else:
                    # to_mix, to_rw = True, True
                    train_loss = self.train_reweight_mixup(ep)

            dev_metric = self.evaluate(self.dev_loader)
            if dev_metric['f'] > best_dev_metric.get('f', 0):
                best_dev_metric = dev_metric
                test_metric = self.evaluate(self.test_loader)
                if test_metric['f'] > best_test_metric.get('f', 0):
                    best_test_metric = test_metric
                    self.save_states(self.args.model_chkp, best_test_metric)
                patient = 0
            else:
                patient += 1

            if patient >= self.args.patient:
                if not to_mix and two_stage:
                    to_mix = True
                    two_stage = False
                    patient = 0
                    self.restore_states(self.args.model_chkp)
                else:
                    break

            logger.info('[Epoch %d] train loss: %.4f, patient: %d, dev_metric: %s, test_metric: %s' %
                        (ep, train_loss, patient, best_dev_metric, best_test_metric))

        test_metric = self.evaluate(self.test_loader)
        if test_metric['f'] > best_test_metric.get('f', 0):
            best_test_metric = test_metric
            self.save_states(self.args.model_chkp, best_test_metric)
        logger.info('Final Dev Metric: %s, Test Metric: %s' % (best_dev_metric, best_test_metric))
        return best_test_metric

    def evaluate(self, test_loader):
        test_pred_tags = []
        test_gold_tags = []
        self.model.eval()
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                pred_score = self.model(batch.bert_inp, batch.mask)
                pred_tag_ids = self.model.tag_decode(pred_score, batch.mask)
                seq_lens = batch.mask.sum(dim=1).tolist()
                for j, l in enumerate(seq_lens):
                    pred_tags = self.vocabs['ner'].idx2inst(pred_tag_ids.cpu()[j][1:l].tolist())
                    gold_tags = batcher[j].ner_tags
                    test_pred_tags.extend(pred_tags)
                    test_gold_tags.extend(gold_tags)
                    assert len(test_gold_tags) == len(test_pred_tags)

        p, r, f = evaluate(test_gold_tags, test_pred_tags, verbose=False)
        return dict(p=p, r=r, f=f)


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():    
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())
    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    data_path = data_config('config/data_path.json')

    random_seeds = [1357, 2789, 3391, 4553, 5919]
    final_res = {'p': [], 'r': [], 'f': []}
    for seed in random_seeds:
        set_seeds(seed)
        trainer = Trainer(args, data_path)
        prf = trainer.run()
        final_res['p'].append(prf['p'])
        final_res['r'].append(prf['r'])
        final_res['f'].append(prf['f'])

    logger.info('Final Result: %s' % final_res)
    final_p = sum(final_res['p']) / len(final_res['p'])
    final_r = sum(final_res['r']) / len(final_res['r'])
    final_f = sum(final_res['f']) / len(final_res['f'])
    logger.info('Final P: %.4f, R: %.4f, F: %.4f' % (final_p, final_r, final_f))
