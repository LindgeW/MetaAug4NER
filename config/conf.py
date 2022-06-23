import os
import json
import argparse


def data_config(data_path):
    assert os.path.exists(data_path)
    with open(data_path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    print(opts)
    return opts


def args_config():
    parse = argparse.ArgumentParser('Parameter Configuration')
    parse.add_argument('--cuda', type=int, default=-1, help='cuda device, default cpu')
    parse.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate of training')
    parse.add_argument('-bt1', '--beta1', type=float, default=0.9, help='beta1 of Adam optimizer 0.9')
    parse.add_argument('-bt2', '--beta2', type=float, default=0.99, help='beta2 of Adam optimizer 0.999')
    parse.add_argument('-eps', '--eps', type=float, default=1e-8, help='eps of Adam optimizer 1e-8')
    parse.add_argument('-warmup', '--warmup_step', type=int, default=10000, help='warm up steps for optimizer')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for Adam optimizer')
    parse.add_argument('--scheduler', choices=['cosine', 'inv_sqrt', 'exponent', 'linear', 'step', 'const'], default='linear', help='the type of lr scheduler')
    parse.add_argument('--grad_clip', type=float, default=5., help='the max norm of gradient clip')
    parse.add_argument('--bert_grad_clip', type=float, default=1., help='the max norm of gradient clip')
    parse.add_argument('--patient', type=int, default=3, help='patient number in early stopping')

    parse.add_argument('--mix_alpha', type=float, default=7, help='mixup parameters')
    parse.add_argument('--aug_lambda', type=float, default=1.0, help='the weight of augmenting loss')
    parse.add_argument('--mixup_lambda', type=float, default=1.0, help='the weight of mixup loss')
    parse.add_argument('--batch_size', type=int, default=16, help='batch size of source inputs')
    parse.add_argument('--aug_batch_size', type=int, default=32, help='batch size of augmentation dataset, should be larger than batch size')
    parse.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parse.add_argument('--epoch', type=int, default=20, help='number of training')
    parse.add_argument('--update_step', type=int, default=1, help='gradient accumulation and update per x steps')

    parse.add_argument('--train_type', choices=['vanilla', 'aug'], default='vanilla', help='the type of train data')
    parse.add_argument('--genre', type=str, help='the type of training data')
    parse.add_argument('--aug_genre', type=str, help='the type of augmented training data')
    parse.add_argument('--to_mix', action='store_true', default=False, help='to do mixup or not')
    parse.add_argument('--to_rw', action='store_true', default=False, help='to do reweight or not')
    parse.add_argument('--two_stage', action='store_true', default=False, help='to do with two stage or not')

    parse.add_argument("--bert_lr", type=float, default=2e-5, help='bert learning rate')
    parse.add_argument("--bert_layer", type=int, default=8, help='the number of last bert layers')
    parse.add_argument('--bert_embed_dim', type=int, default=768, help='feature size of bert inputs')
    parse.add_argument('--hidden_size', type=int, default=400, help='feature size of hidden layer')
    parse.add_argument('--rnn_depth', type=int, default=1, help='number of rnn layers')
    parse.add_argument('--embed_drop', type=float, default=0.5, help='drop rate of embedding layer')
    parse.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')

    parse.add_argument('--model_chkp', type=str, default='model.pkl', help='model saving path')
    parse.add_argument('--vocab_chkp', type=str, default='vocab.pkl', help='vocab saving path')

    args = parse.parse_args()

    print(vars(args))

    return args



