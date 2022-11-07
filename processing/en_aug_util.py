import itertools
import random
import sys
from synonym import synonyms


class Instance:
    def __init__(self, wd, ner):
        self.wd = wd
        self.ner = ner

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


def load_dic(path):
    ner_dic = dict()
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                tp, ents = line.split('|||')
                ner_dic[tp.strip()] = ents.strip().split(' ')
    return ner_dic


def load_data(path):
    sent_inst = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < 2:
                if len(sent_inst) > 0:
                    yield sent_inst
                sent_inst = []
            else:
                sent_inst.append(Instance(*tokens))
    if len(sent_inst) > 0:
        yield sent_inst


def read_data_from(path):
    data_set = []
    for insts in load_data(path):
        data_set.append(insts)
    return data_set


def entity_replace(insts, ner_dic, ratio=0.5):
    ners = []
    one_ent = []
    start = False
    for i, inst in enumerate(insts):
        if inst.ner == 'O' or inst.ner.startswith('S-'):
            start = False
            one_ent = []
            ners.append([inst])
            continue
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

    for i, items in enumerate(ners):
        if items[0].ner != 'O' and random.random() < ratio:
            tp = items[0].ner.split('-')[1]
            new_ent = random.sample(ner_dic[tp], 1)[0]
            new_ents = new_ent.split(chr(1))
            new_inst = []
            for j, w in enumerate(new_ents):
                if len(new_ents) == 1:
                    ner_tag = 'S-' + tp
                elif j == 0:
                    ner_tag = 'B-' + tp
                elif j == len(new_ents) - 1:
                    ner_tag = 'E-' + tp
                else:
                    ner_tag = 'I-' + tp
                new_inst.append(Instance(w, ner_tag))
            ners[i] = new_inst

    return list(itertools.chain.from_iterable(ners))


def nonentity_replace(insts, ratio=0.5):
    for i, inst in enumerate(insts):
        if inst.ner.upper() == 'O':
            wds = synonyms.nearby(inst.wd)
            if wds is not None and random.random() < ratio:
                insts[i].wd = random.choice(wds)
    return insts


def do_aug(data_path, save_path, dic_path, ratio=0.5, aug_times=1):
    ner_dic = load_dic(dic_path)
    dataset = read_data_from(data_path)
    new_insts = []
    for _ in range(aug_times):
        for i, sent in enumerate(dataset):
            sent_inst = entity_replace(sent, ner_dic, ratio)
            sent_inst = nonentity_replace(sent_inst, 1 - ratio)
            if sent_inst:
                new_insts.append(sent_inst)

    if len(new_insts) == 0:
        new_insts = dataset

    with open(save_path, 'w', encoding='utf-8') as fw:
        for insts in new_insts:
            for inst in insts:
                fw.write(f'{inst.wd}\t{inst.ner}\n')
            fw.write('\n')
    print('Done!')


# python aug_utils.py onto5.train train.repl1 train.dic 0.5 1
def run():
    random.seed(1234)
    train_file, out_file, dic_file, ratio, times = sys.argv[1:6]
    do_aug(train_file,
           out_file,
           dic_file,
           ratio=float(ratio),
           aug_times=int(times))


run()
