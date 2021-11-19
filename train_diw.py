import os
import time
import torch
import random
import numpy as np
from modules.optimizer import Optimizer, AdamW, WarmupLinearSchedule
# from model.seq_tagger import BertSeqTagger
from model.mix_seq_tagger import BertSeqTagger
#from model.mix_seq_tagger2 import BertSeqTagger
from config.conf import args_config, data_config
from utils.dataset import DataLoader, BucketDataLoader, BatchWrapper
from utils.datautil import load_data, create_vocab, batch_variable, save_to
from utils.conlleval import evaluate
from utils.tag_util import *
from utils.eval import calc_prf
import torch.nn.utils as nn_utils
from logger.logger import logger
import torch.nn.functional as F


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

        self.vocabs = create_vocab(self.train_set, data_config['pretrained']['bert_model'], embed_file=None)
        #save_to(args.vocab_chkp, self.vocabs)
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
        print("Training %dM trainable parameters..." % (total_params/1e6))

        self.train_loader = BatchWrapper(BucketDataLoader(self.train_set, batch_size=self.args.batch_size, key=lambda x: len(x.chars), shuffle=True, sort_within_batch=True), mixup=False)
        if self.aug_train_set is not None:
            aug_bs = self.args.aug_batch_size
            #aug_bs = int(self.args.batch_size * len(self.aug_train_set) / len(self.train_set))
            print('augmented batch size:', aug_bs)
            self.aug_train_loader = BatchWrapper(BucketDataLoader(self.aug_train_set, batch_size=aug_bs, key=lambda x: len(x.chars), shuffle=True, sort_within_batch=True), mixup=False)
            # nb_aug_batch = len(aug_train_loader)
        self.args.max_step = self.args.epoch * (len(self.train_loader) // self.args.update_step)

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.bert_named_params()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        self.bert_optimizer = AdamW(self.optimizer_bert_parameters, lr=self.args.bert_lr, eps=1e-8)
        self.bert_scheduler = WarmupLinearSchedule(self.bert_optimizer, warmup_steps=len(self.train_loader)//self.args.update_step, t_total=self.args.max_step)
        self.base_params = self.model.non_bert_params()
        self.optimizer = Optimizer(self.base_params, self.args)

    def calc_train_acc(self, pred_score, gold_tags, mask=None):
        '''
        :param pred_score: (b, t, nb_tag)
        :param gold_tags: (b, t)
        :param mask: (b, t) 1对于有效部分，0对应pad
        :return:
        '''
        pred_tags = pred_score.data.argmax(dim=-1)
        nb_right = ((pred_tags == gold_tags) * mask).sum().item()
        nb_total = mask.sum().item()
        return nb_right, nb_total

    # ner eval
    def eval_ner(self, pred_tag_ids, gold_tag_ids, mask, ner_vocab, return_prf=False):
        '''
        :param pred_tag_ids: (b, t)
        :param gold_tag_ids: (b, t)
        :param mask: (b, t)  0 for padding
        :return:
        '''
        seq_lens = mask.sum(dim=1).tolist()
        nb_right, nb_pred, nb_gold = 0, 0, 0
        for i, l in enumerate(seq_lens):
            pred_tags = ner_vocab.idx2inst(pred_tag_ids[i][1:l].tolist())
            gold_tags = ner_vocab.idx2inst(gold_tag_ids[i][1:l].tolist())
            pred_ner_spans = set(extract_ner_bieso_span(pred_tags))
            gold_ner_spans = set(extract_ner_bieso_span(gold_tags))
            nb_pred += len(pred_ner_spans)
            nb_gold += len(gold_ner_spans)
            nb_right += len(pred_ner_spans & gold_ner_spans)

        if return_prf:
            return calc_prf(nb_right, nb_pred, nb_gold)
        else:
            return nb_right, nb_pred, nb_gold

    # vanilla training
    def train_eval(self, ep=0, mixup=False):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_right, train_total = 0, 0
        for i, batch_train_data in enumerate(self.train_loader):
            if mixup:
                batcher, mix_lmbd, batcher2, _ = batch_train_data
                batch = batch_variable(batcher, self.vocabs)
                batch2 = batch_variable(batcher2, self.vocabs)
                batch.to_device(self.args.device)
                batch2.to_device(self.args.device)
                mix_lmbd = mix_lmbd.to(self.args.device)

                tag_score, inst_w = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_lmbd)
                loss = self.model.tag_loss(tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                           penalty_ws=mix_lmbd)
                loss += self.model.tag_loss(tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                            penalty_ws=1 - mix_lmbd)
            else:
                batcher, _ = batch_train_data
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                tag_score, _ = self.model(batch.bert_inp, batch.mask)
                #adv_loss = - torch.mean(torch.log(inst_w))  # gold labels equal 1
                loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
                #loss += adv_loss

            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == len(self.train_loader)):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
            print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])
        return train_loss


    # clean data & pseudo data interative training with adv weight
    def train_eval_with_aug(self, ep=1):
        best_dev_metric, best_test_metric = dict(), dict()
        patient = 0
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_right, train_total = 0, 0
        for i, ((batcher, mix_lmbd), (aug_batcher, _)) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            batch = batch_variable(batcher, self.vocabs)
            batch.to_device(self.args.device)
            aug_batch = batch_variable(aug_batcher, self.vocabs)
            aug_batch.to_device(self.args.device)
        
            tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
            loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask, reduction='mean')
            inst_loss = - torch.mean(torch.log(inst_w))
            loss += inst_loss

            aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
            #aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, reduction='mean')
            aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w, reduction='mean')
            loss += aug_loss

            print('clean inst weight:', inst_w.data.tolist())
            print('aug inst weight:', aug_inst_w.data.tolist())
            #print([''.join(x.chars) for x in aug_batcher[:5]])

            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == len(self.train_loader)):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
            print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])
        return train_loss


    # add aug data and mix_up for training - 统一成整体
    def train_eval_with_aug_mixup(self, ep, mixup=False):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        if mixup:
            self.train_loader.set_mixup(True)
            self.aug_train_loader.set_mixup(True)
            print('clean data mixup training .....')

        for i, (batch_train_data, batch_aug_data) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            if mixup:
                # mix clean data
                batcher, mix_lmbd, batcher2, _ = batch_train_data
                batch = batch_variable(batcher, self.vocabs)
                batch2 = batch_variable(batcher2, self.vocabs)
                batch.to_device(self.args.device)
                batch2.to_device(self.args.device)
                mix_lmbd = mix_lmbd.to(self.args.device)

                tag_score, mix_inst_w = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_lmbd)
                #mix_adv_loss = - torch.mean(torch.log(mix_inst_w))
                loss = self.model.tag_loss(tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask, penalty_ws=mix_lmbd)
                loss += self.model.tag_loss(tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask, penalty_ws=(1 - mix_lmbd))
                #loss += mix_adv_loss
            else:
                # clean data
                batcher, _ = batch_train_data
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
                adv_loss = - torch.mean(torch.log(inst_w))  # gold labels equal 1
                loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
                loss += adv_loss

            '''            
            # aug data
            aug_batcher, _ = batch_aug_data
            aug_batch = batch_variable(aug_batcher, self.vocabs)
            aug_batch.to_device(self.args.device)
            aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
            aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w)
            loss += self.args.aug_lambda * aug_loss
            '''
            aug_batcher, aug_mix_lmbd, aug_batcher2, _ = batch_aug_data
            aug_batch = batch_variable(aug_batcher, self.vocabs)
            aug_batch2 = batch_variable(aug_batcher2, self.vocabs)
            aug_batch.to_device(self.args.device)
            aug_batch2.to_device(self.args.device)
            aug_mix_lmbd = aug_mix_lmbd.to(self.args.device)
            aug_tag_score, aug_mix_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask, aug_batch2.bert_inp, aug_batch2.mask, aug_mix_lmbd)
            #loss += self.model.tag_loss(aug_tag_score[:, :aug_batch.ner_ids.shape[1]], aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_mix_lmbd)
            #loss += self.model.tag_loss(aug_tag_score[:, :aug_batch2.ner_ids.shape[1]], aug_batch2.ner_ids, aug_batch2.mask, penalty_ws=1-aug_mix_lmbd)        
            loss += self.model.tag_loss(aug_tag_score[:, :aug_batch.ner_ids.shape[1]], aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_mix_lmbd, mixup_ws=aug_mix_inst_w)
            loss += self.model.tag_loss(aug_tag_score[:, :aug_batch2.ner_ids.shape[1]], aug_batch2.ner_ids, aug_batch2.mask, penalty_ws=1-aug_mix_lmbd, mixup_ws=aug_mix_inst_w)        
            

            loss_val = loss.data.item()
            train_loss += loss_val

            # mix_tag_score, _ = self.model(batch.bert_inp, batch.mask, aug_batch.bert_inp, aug_batch.mask, mix_lmbd)
            # mix_loss = self.model.tag_loss(mix_tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
            #                                penalty_ws=mix_lmbd)
            # mix_loss += self.model.tag_loss(mix_tag_score[:, :aug_batch.ner_ids.shape[1]], aug_batch.ner_ids,
            #                                 aug_batch.mask, penalty_ws=1 - mix_lmbd)
            # loss += mix_loss * self.args.mixup_lambda

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == len(self.train_loader)):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))

        return train_loss
 
    def train_with_aug_mixup_iter0(self, ep):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_right, train_total = 0, 0
        self.aug_train_loader.set_mixup(True)
        N = len(self.train_loader)
        for i, ((batcher, mix_lmbd), (aug_batcher, _, aug_batcher2, _)) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            if len(batcher) != len(aug_batcher2):
                continue
            batch = batch_variable(batcher, self.vocabs)
            aug_batch = batch_variable(aug_batcher, self.vocabs, stat='aug')
            aug_batch2 = batch_variable(aug_batcher2, self.vocabs, stat='aug')
            batch.to_device(self.args.device)
            aug_batch.to_device(self.args.device)
            aug_batch2.to_device(self.args.device)
            mix_lmbd = mix_lmbd.to(self.args.device)
        
            loss = 0.
            if i <= N // 2:
            #if i < 0:
                # clear data
                tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
                loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
                loss += - torch.mean(torch.log(inst_w))
                # aug data
                aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
                aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w)
                loss += self.args.aug_lambda * aug_loss
   
            # clean mixup
            mix_tag_score, _ = self.model(batch.bert_inp, batch.mask, aug_batch2.bert_inp, aug_batch2.mask, mix_lmbd)
            #mix_adv_loss = - torch.mean(torch.log(mix_inst_w))
            loss += self.model.tag_loss(mix_tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                            penalty_ws=mix_lmbd)
            loss += self.model.tag_loss(mix_tag_score[:, :aug_batch2.ner_ids.shape[1]], aug_batch2.ner_ids, aug_batch2.mask,
                                            penalty_ws=1 - mix_lmbd)
            #loss += mix_adv_loss
            
            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == N):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
            print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])
        return train_loss
    
    def train_with_aug_mixup_iter1(self, ep):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_right, train_total = 0, 0
        self.train_loader.set_batch_size(self.args.batch_size * 2)
        self.train_loader.set_mixup(True)
        N = len(self.train_loader)
        for i, ((batcher, mix_lmbd, batcher2, _), (aug_batcher, _)) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            batch = batch_variable(batcher, self.vocabs)
            aug_batch = batch_variable(aug_batcher, self.vocabs, stat='aug')
            batch.to_device(self.args.device)
            aug_batch.to_device(self.args.device)
            mix_lmbd = mix_lmbd.to(self.args.device)
        
            loss = 0.
            if i <= N // 2:
                # clear data
                tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
                loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
                loss += - torch.mean(torch.log(inst_w))
                # aug data
                aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
                aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w)
                loss += aug_loss
            else:
                #sample_batcher = np.random.choice(batcher2+aug_batcher, len(batcher), replace=False)
                #batch2 = batch_variable(sample_batcher, self.vocabs)
                batch2 = batch_variable(batcher2, self.vocabs)
                batch2.to_device(self.args.device)
                # clean mixup
                mix_tag_score, mix_inst_w = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_lmbd)
                loss = self.model.tag_loss(mix_tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                            penalty_ws=mix_lmbd, mixup_ws=mix_inst_w)
                loss += self.model.tag_loss(mix_tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                            penalty_ws=1 - mix_lmbd, mixup_ws=mix_inst_w)
                
                aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
                aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w)
                #loss += self.args.aug_lambda * aug_loss + mix_adv_loss
                loss += self.args.aug_lambda * aug_loss
            
            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == N):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
            print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])
        return train_loss

    # clean mixup + pesudo aug
    def train_with_aug_mixup_iter2(self, ep):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_right, train_total = 0, 0
        self.train_loader.set_batch_size(self.args.batch_size * 2)
        self.train_loader.set_mixup(True)
        N = len(self.train_loader)
        for i, ((batcher, mix_lmbd, batcher2, _), (aug_batcher, _)) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            batch = batch_variable(batcher, self.vocabs)
            batch2 = batch_variable(batcher2, self.vocabs)
            aug_batch = batch_variable(aug_batcher, self.vocabs, stat='aug')
            batch.to_device(self.args.device)
            batch2.to_device(self.args.device)
            aug_batch.to_device(self.args.device)
            mix_lmbd = mix_lmbd.to(self.args.device)
        
            # clear data
            tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
            loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
            loss += - torch.mean(torch.log(inst_w))
            # aug data
            aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
            aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w)
            loss += self.args.aug_lambda * aug_loss

            if i >= N // 2:
                # clean mixup
                mix_tag_score, _ = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_lmbd)
                #mix_adv_loss = - torch.mean(torch.log(mix_inst_w))
                loss += self.model.tag_loss(mix_tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                           penalty_ws=mix_lmbd)
                loss += self.model.tag_loss(mix_tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                            penalty_ws=1 - mix_lmbd)
                #loss += mix_adv_loss
            
            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()


            if (i + 1) % self.args.update_step == 0 or (i + 1 == N):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
            print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])
        return train_loss

    def train_with_aug_mixup_iter3(self, ep):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        train_right, train_total = 0, 0
        self.train_loader.set_batch_size(self.args.batch_size * 2)
        self.train_loader.set_mixup(True)
        N = len(self.train_loader)
        for i, ((batcher, mix_lmbd, batcher2, _), (aug_batcher, _)) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            batch = batch_variable(batcher, self.vocabs)
            batch2 = batch_variable(batcher2, self.vocabs)
            aug_batch = batch_variable(aug_batcher, self.vocabs, stat='aug')
            batch.to_device(self.args.device)
            batch2.to_device(self.args.device)
            aug_batch.to_device(self.args.device)
            mix_lmbd = mix_lmbd.to(self.args.device)
        
            # clear data
            tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
            loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
            loss += - torch.mean(torch.log(inst_w))
            # aug data
            aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
            aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask, penalty_ws=aug_inst_w)
            loss += self.args.aug_lambda * aug_loss
                
            # clean mixup
            mix_tag_score, _ = self.model(batch.bert_inp, batch.mask, batch2.bert_inp, batch2.mask, mix_lmbd)
            #mix_adv_loss = - torch.mean(torch.log(mix_inst_w))
            loss += self.model.tag_loss(mix_tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask,
                                       penalty_ws=mix_lmbd)
            loss += self.model.tag_loss(mix_tag_score[:, :batch2.ner_ids.shape[1]], batch2.ner_ids, batch2.mask,
                                        penalty_ws=1 - mix_lmbd)
            #loss += mix_adv_loss
            
            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == N):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
            print('bert lr:', self.bert_optimizer.param_groups[0]['lr'])
        return train_loss

    def train_epoch_wise(self, ep):
        self.model.train()
        t1 = time.time()
        train_loss = 0.
        N = len(self.train_loader)
        for i, (batch_train_data, batch_aug_data) in enumerate(zip(self.train_loader, self.aug_train_loader)):
            batcher, mix_lmbd = batch_train_data
            aug_batcher, _ = batch_aug_data
            batch = batch_variable(batcher, self.vocabs)
            batch.to_device(self.args.device)
            mix_lmbd = mix_lmbd.to(self.args.device)

            if i < N // 3:   # clean data warm-up
                tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
                adv_loss = - torch.mean(torch.log(inst_w))  # gold labels equal 1
                loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
                loss += adv_loss
            elif i < 2 * N // 3:
                aug_batch = batch_variable(aug_batcher, self.vocabs, stat='aug')
                aug_batch.to_device(self.args.device)
                
                # clear data
                tag_score, inst_w = self.model(batch.bert_inp, batch.mask)
                adv_loss = - torch.mean(torch.log(inst_w))
                loss = self.model.tag_loss(tag_score, batch.ner_ids, batch.mask)
                loss += adv_loss
                # aug data
                aug_tag_score, aug_inst_w = self.model(aug_batch.bert_inp, aug_batch.mask)
                aug_loss = self.model.tag_loss(aug_tag_score, aug_batch.ner_ids, aug_batch.mask,  penalty_ws=aug_inst_w)
                loss += self.args.aug_lambda * aug_loss
            else:
                sample_aug_batcher = np.random.choice(aug_batcher, len(batcher), replace=False)
                aug_batch = batch_variable(sample_aug_batcher, self.vocabs, stat='aug')
                aug_batch.to_device(self.args.device)
                
                mix_tag_score, _ = self.model(batch.bert_inp, batch.mask, aug_batch.bert_inp, aug_batch.mask, mix_lmbd)
                # mix_adv_loss = - torch.mean(torch.log(mix_inst_w))
                loss = self.model.tag_loss(mix_tag_score[:, :batch.ner_ids.shape[1]], batch.ner_ids, batch.mask, penalty_ws=mix_lmbd)
                loss += self.model.tag_loss(mix_tag_score[:, :aug_batch.ner_ids.shape[1]], aug_batch.ner_ids, aug_batch.mask, penalty_ws=1 - mix_lmbd)
                # loss *= self.args.mixup_lambda
                # loss += mix_adv_loss

            loss_val = loss.data.item()
            train_loss += loss_val

            if self.args.update_step > 1:
                loss = loss / self.args.update_step

            loss.backward()

            if (i + 1) % self.args.update_step == 0 or (i + 1 == N):
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.base_params),
                                         max_norm=self.args.grad_clip)
                nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()),
                                         max_norm=self.args.bert_grad_clip)
                self.optimizer.step()
                self.bert_optimizer.step()
                self.bert_scheduler.step()
                self.model.zero_grad()

            logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f' % (
                ep, i, (time.time() - t1), self.optimizer.get_lr(), loss_val))
        return train_loss

    def save_states(self, save_path, best_test_metric=None):
        bert_optim_state = {'optimizer': self.bert_optimizer.state_dict(), 
                            'scheduler': self.bert_scheduler.state_dict()
                           }
        check_point = {'best_prf': best_test_metric,
                       'model_state': self.model.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'bert_optimizer_state': bert_optim_state,
                       'args_settings': self.args,
                       #'rng_state': torch.get_rng_state(),  # 随机生成器状态（Byte Tensor）
                       }
        torch.save(check_point, save_path)
        logger.info(f'Saved the current model states to {save_path} ...')

    def restore_states(self, load_path):
        ckpt = torch.load(load_path)
        # torch.set_rng_state(ckpt['rng_state'])
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.bert_optimizer.load_state_dict(ckpt['bert_optimizer_state']['optimizer'])
        self.bert_scheduler.load_state_dict(ckpt['bert_optimizer_state']['scheduler'])
        #self.bert_optimizer.param_groups[0]['lr'] = 2e-5
        #self.bert_scheduler = WarmupLinearSchedule(self.bert_optimizer, warmup_steps=0, t_total=self.args.max_step)
        self.args = ckpt['args_settings']
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        self.bert_optimizer.zero_grad()
        logger.info('Loading the previous model states ...')
        print('Previous best prf result is: %s' % ckpt['best_prf'])

  
    # vanilla -> aug data -> mixup
    def train(self):
        patient = 0
        best_dev_metric, best_test_metric = dict(p=0,r=0,f=0), dict(p=0,r=0,f=0)
        stage_flag = 'vanilla'
        #stage_flag = 'aug'
        mixup = False
        for ep in range(1, 1+self.args.epoch):
            if self.aug_train_set is not None and stage_flag == 'aug':
                print('training for aug .....')
                train_loss = self.train_eval_with_aug(ep)

                #train_loss = self.train_with_aug_mixup_iter0(ep)
                #train_loss = self.train_with_aug_mixup_iter1(ep)
                #train_loss = self.train_with_aug_mixup_iter2(ep)
                #train_loss = self.train_with_aug_mixup_iter3(ep)
            elif self.aug_train_set is not None and stage_flag == 'mixup':
                train_loss = self.train_eval_with_aug_mixup(ep, mixup=True)
            else:
                print('training for vanilla .....')
                train_loss = self.train_eval(ep, mixup)

            dev_metric = self.evaluate(self.val_set)
            if dev_metric['f'] > best_dev_metric.get('f', 0):
                best_dev_metric = dev_metric
                test_metric = self.evaluate(self.test_set)
                if test_metric['f'] > best_test_metric.get('f', 0):
                    best_test_metric = test_metric
                    self.save_states(self.args.model_chkp, best_test_metric)
                patient = 0
            else:
                patient += 1

            #if patient >= self.args.patient or (stage_flag=='vanilla' and ep % 15 == 0):
            if patient >= self.args.patient:
                if self.args.train_type == 'vanilla':
                    test_metric = self.evaluate(self.test_set)
                    if test_metric['f'] > best_test_metric.get('f', 0):
                        best_test_metric = test_metric
                        self.save_states(self.args.model_chkp, best_test_metric)
                        print('vanilla stage test metric: %s' % best_test_metric)
                    break

                if stage_flag == 'vanilla':
                    test_metric = self.evaluate(self.test_set)
                    if test_metric['f'] > best_test_metric.get('f', 0):
                        best_test_metric = test_metric
                        self.save_states(self.args.model_chkp, best_test_metric)
                        print('first stage test metric: %s' % best_test_metric)

                    #stage_flag = 'mixup'
                    stage_flag = 'aug'
                    patient = 0
                    self.restore_states(self.args.model_chkp)
                    print('change to pseudo data training ...')
                elif stage_flag == 'aug':
                    test_metric = self.evaluate(self.test_set)
                    if test_metric['f'] > best_test_metric.get('f', 0):
                        best_test_metric = test_metric
                        self.save_states(self.args.model_chkp, best_test_metric)
                        print('stage test metric: %s' % best_test_metric)
                    
                    stage_flag = 'mixup'
                    mixup = True
                    patient = 0
                    #self.train_loader.set_mixup(True)
                    self.restore_states(self.args.model_chkp)
                    print('chang to clean mixup data training ...')
                    #break
                else:
                    test_metric = self.evaluate(self.test_set)
                    if test_metric['f'] > best_test_metric.get('f', 0):
                        best_test_metric = test_metric
                        self.save_states(self.args.model_chkp, best_test_metric)
                        print('stage test metric: %s' % best_test_metric)
                    break

            logger.info('[Epoch %d] train loss: %.4f, lr: %f, patient: %d, dev_metric: %s, test_metric: %s' %
                        (ep, train_loss, self.optimizer.get_lr(), patient, best_dev_metric, best_test_metric))

        logger.info('Final Dev Metric: %s, Test Metric: %s' % (best_dev_metric, best_test_metric))
        return best_test_metric

    def evaluate(self, test_data):
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False)
        test_pred_tags = []
        test_gold_tags = []
        ner_vocab = self.vocabs['ner']
        self.model.eval()
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)
                mask = batch.mask
                pred_score, _ = self.model(batch.bert_inp, mask)
                pred_tag_ids = self.model.tag_decode(pred_score, mask)
                seq_lens = batch.mask.sum(dim=1).tolist()
                for j, l in enumerate(seq_lens):
                    pred_tags = ner_vocab.idx2inst(pred_tag_ids[j][1:l].tolist())
                    #gold_tags = ner_vocab.idx2inst(batch.ner_ids[j][1:l].tolist())
                    gold_tags = batcher[j].ner_tags
                    test_pred_tags.extend(pred_tags)
                    test_gold_tags.extend(gold_tags)
                    assert len(test_gold_tags) == len(test_pred_tags)

        p, r, f = evaluate(test_gold_tags, test_pred_tags, verbose=False)
        return dict(p=p, r=r, f=f)


def set_seeds(seed=3349):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    data_path = data_config('./config/small_data_path.json')

    #train_set = load_data(data_path['wb']['train'])
    #vocabs = create_vocab(train_set, data_path['pretrained']['bert_model'], embed_file=None)
    #print(len(vocabs['ner']))
    #save_to(args.vocab_chkp, vocabs)
    
    random_seeds = [1357, 2789, 3391, 4553, 5919]
    final_res = {'p': [], 'r': [], 'f': []}
    for seed in random_seeds:
        set_seeds(seed)
        trainer = Trainer(args, data_path)
        prf=trainer.train()
        #prf=trainer.train_aug()
        #prf=trainer.train_eval_with_aug_mixup()
        #prf=trainer.train_eval_with_aug_mixup_iter()
       
        final_res['p'].append(prf['p'])
        final_res['r'].append(prf['r'])
        final_res['f'].append(prf['f'])
    
    logger.info('Final Result: %s' % final_res)
    final_p = sum(final_res['p']) / 5
    final_r = sum(final_res['r']) / 5
    final_f = sum(final_res['f']) / 5
    logger.info('Final P: %.4f, R: %.4f, F: %.4f' % (final_p, final_r, final_f))
    print()


