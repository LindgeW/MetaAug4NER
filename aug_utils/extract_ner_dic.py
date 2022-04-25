import os
import sys
from collections import Counter

class Instance:
    def __init__(self, wd, ner):
        self.wd = wd
        self.ner = ner

    def __repr__(self):
        return str(self.__dict__)


def read_file(path):
    assert os.path.exists(path)
    sents = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < 2:
                if len(sents) > 0:
                    yield sents
                sents = []
            else:
                sents.append(Instance(*tokens))
    if len(sents) > 0:
        yield sents


# O O O B-LOC M-LOC O O B-PER E-PER M-PER O O
def get_ners(insts):
    ners = []
    one_ent = []
    start = False
    for i, inst in enumerate(insts):
        if inst.ner == 'O':
            start = False
            one_ent = []
            continue
        elif inst.ner.startswith('S-'):
            ners.append([inst])
            start = False
            one_ent = []
        elif inst.ner.startswith('B-'):
            one_ent = [inst]
            start = True
        elif inst.ner.startswith('E-'):
            if start and one_ent:
                one_ent.append(inst)
                ners.append(one_ent)
                one_ent = []
                start = False
        else:
            if inst.ner.startswith('M-') or inst.ner.startswith('I-'):
                one_ent.append(inst)
    return ners


def extract_ners(in_path, out_path):
    all_insts = read_file(in_path)
    fw = open(out_path, 'w', encoding='utf-8')
    ent_ctr = dict()
    #ent_wd_counter = Counter()
    for insts in all_insts:
        sent_ners = get_ners(insts)
        for ners in sent_ners:
            tp = ners[0].ner.split('-')[1]
            wd = '_'.join([ner.wd for ner in ners])
            if len(ent_ctr) > 0 and tp in ent_ctr:
                ent_ctr[tp].add(wd)
                #ent_wd_counter[(tp, wd)] += 1
            else:
                #ent_wd_counter[(tp, wd)] += 1
                ent_ctr[tp] = set([wd])

    for tp, wds in ent_ctr.items():
        fw.write(f'{tp} ||| {" ".join(wds)}\n')
    fw.close()


def run():
    train_file, out_dic_file = sys.argv[1:3]
    extract_ners(train_file, out_dic_file)

run()


