import os
from utils.instance import Instance
from utils.vocab import Vocab, MultiVocab, BERTVocab
import torch
import collections
import pickle
import numpy as np


# def read_insts(file_reader):
#     insts = []
#     for line in file_reader:
#         try:
#             tokens = line.strip().split()
#             if line.strip() == '' or len(tokens) < 3:
#                 if len(insts) > 0:
#                     yield insts
#                 insts = []
#             elif len(tokens) == 3:
#                 insts.append(Instance(tokens[0], tokens[1], tokens[2]))
#         except Exception as e:
#             print('exception occur: ', e)
#
#     if len(insts) > 0:
#         yield insts


def read_insts(file_reader):
    new_inst = {'chars': [], 'ner_tags': []}
    num_col = 2
    for line in file_reader:
        try:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < num_col:
                if len(new_inst['chars']) > 0:
                    yield Instance(**new_inst)
                new_inst = {'chars': [], 'ner_tags': []}
            elif len(tokens) == num_col:
                new_inst['chars'].append(tokens[0])
                new_inst['ner_tags'].append(tokens[-1])
        except Exception as e:
            print('exception occur: ', e)

    if len(new_inst['chars']) > 0:
        yield Instance(**new_inst)


def load_data(path):
    assert os.path.exists(path)
    dataset = []
    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_insts(fr):
            if len(inst.chars) < 512:
                dataset.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset


def create_vocab(data_sets, bert_model_path=None, embed_file=None):
    bert_vocab = BERTVocab(bert_model_path)
    ner_tag_vocab = Vocab(unk=None, bos=None, eos=None)
    for inst in data_sets:
        ner_tag_vocab.add(inst.ner_tags)

    # if embed_file is not None:
    #     embed_count = char_vocab.load_embeddings(embed_file)
    #     print("%d word pre-trained embeddings loaded..." % embed_count)

    return MultiVocab(dict(
        ner=ner_tag_vocab,
        bert=bert_vocab
    ))


def _is_chinese(a_chr):
    return u'\u4e00' <= a_chr <= u'\u9fff'


def batch_variable(batch_data, mVocab, stat=None):
    batch_size = len(batch_data)
    max_seq_len = 1 + max(len(inst.chars) for inst in batch_data)  # align [CLS] !!!

    bert_vocab = mVocab['bert']
    ner_tag_vocab = mVocab['ner']
    ner_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    chars = []
    for i, inst in enumerate(batch_data):
        seq_len = len(inst.chars) + 1
        chars.append(inst.chars)
        mask[i, :seq_len].fill_(1)
        ner_ids[i, :seq_len] = torch.tensor([ner_tag_vocab.inst2idx(nt) for nt in ['O'] + inst.ner_tags])

    bert_inps = bert_vocab.batch_bertwd2id(chars)
    return Batch(bert_inp=bert_inps,
                 ner_ids=ner_ids,
                 mask=mask)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
            elif isinstance(val, collections.abc.Sequence) or isinstance(val, collections.abc.Iterable):
                val_ = [v.to(device) if torch.is_tensor(v) else v for v in val]
                setattr(self, prop, val_)
        return self


def save_to(path, obj):
    if os.path.exists(path):
        return None
    with open(path, 'wb') as fw:
        pickle.dump(obj, fw)
    print('Obj saved!')


def load_from(pkl_file):
    with open(pkl_file, 'rb') as fr:
        obj = pickle.load(fr)
    return obj

