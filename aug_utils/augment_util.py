import sys
import os
import random
import itertools
from sim_wd import Synonym
synonyms = Synonym()
#res = synonyms.nearby('sad')


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
                ner_type, wds = line.split('|||')
                ner_dic[ner_type.strip()] = wds.strip().split(' ')
    return ner_dic


def write_to_conll(path, sent_insts):
    nb_sent = 0
    with open(path, 'w', encoding='utf-8') as fw:
        for insts in sent_insts:
            nb_sent += 1
            for ins in insts:
                fw.write(f'{ins.wd}\t{ins.ner}\n')
            fw.write('\n')
    print(nb_sent, 'sents are written to', path)


def read_from_file(path):
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


# 随机采样指定比例的数据
def read_data(path, ratio=1., seed=None):
    if seed:
        random.seed(seed)
    loader = read_from_file(path)
    data_set = []
    for insts in loader:
        data_set.append(insts)

    if ratio < 1:
        n = len(data_set)
        samples = random.sample(data_set, int(ratio*n))
        return samples
    else:
        return data_set

def chk_inst(insts):
    full_O = True
    for ins in insts:
        if ins.ner != 'O':
            full_O = False
            break
    return full_O


def rand_train_dev_split(path, ratio=0.1, seed=None):
    if seed:
        random.seed(seed)
    loader = read_from_file(path)
    dataset = []
    for insts in loader:
        dataset.append(insts)    
    n = len(dataset)
    random.shuffle(dataset)
    n_split = int(n * ratio)
    
    train_set = dataset[n_split:]
    dev_set = dataset[:n_split]
    print(n, len(train_set), len(dev_set))
    return train_set, dev_set
 

# O O O B-LOC M-LOC O O B-PER E-PER M-PER O O
# def replace_one_instance(insts, ner_dic):
#     one_ent = []
#     start = False
#     new_insts = []
#     for i, inst in enumerate(insts):
#         if inst.cws == 'S':
#             start = False
#             one_ent = []
#
#             new_insts.append([inst])
#         elif inst.cws == 'B':
#             one_ent = [inst]
#             start = True
#         elif inst.cws == 'E':
#             if start and one_ent:
#                 one_ent.append(inst)
#                 new_insts.append(one_ent)
#             one_ent = []
#             start = False
#         else:
#             if inst.cws == 'M' or inst.cws == 'I':
#                 one_ent.append(inst)
#
#     for i, items in enumerate(new_insts):
#         if items[0].ner != 'O':
#             ner_type = items[0].ner.split('-')[1]
#             cands = ner_dic[ner_type]
#             new_wd = random.sample(cands, 1)[0]
#             rep_insts = []
#             print(new_wd)
#             for j, w in enumerate(new_wd):
#                 if len(new_wd) < 2:
#                     cws_tag = 'S'
#                     ner_tag = 'S-' + ner_type
#                 elif j == 0:
#                     cws_tag = 'B'
#                     ner_tag = 'B-' + ner_type
#                 elif j == len(new_wd) - 1:
#                     cws_tag = 'E'
#                     ner_tag = 'E-' + ner_type
#                 else:
#                     cws_tag = 'M'
#                     ner_tag = 'M-' + ner_type
#                 rep_insts.append(Instance(w, cws_tag, ner_tag))
#             new_insts[i] = rep_insts
#     return new_insts


def synonyms_replace_bies(insts):
    seg_insts = []
    one_wd = []
    is_start = False
    for i, inst in enumerate(insts):
        # bmes / bies
        if inst.cws.lower() not in 'bmes':
            is_start = False
            continue
        if inst.cws.lower() == 's':
            seg_insts.append([inst])
            is_start = False
        elif inst.cws.lower() == 'b':
            one_wd = [inst]
            is_start = True
        elif inst.cws.lower() == 'm' or inst.cws.lower() == 'i':
            if is_start:
                one_wd.append(inst)
        elif inst.cws.lower() == 'e':
            if is_start:
                one_wd.append(inst)
                seg_insts.append(one_wd)
                is_start = False
            one_wd = []

    new_insts = []
    for ins in seg_insts:
        if ins[0].ner != 'O':
            new_insts.extend(ins)
        else:
            wds, _ = synonyms.nearby(''.join([s.wd for s in ins]))
            if wds and random.random() < 0.2:
                syn_wd = random.choice(wds[1:])
                for j, swd in enumerate(syn_wd):
                    if len(syn_wd) == 1:
                        cws = 'S'
                    elif j == 0:
                        cws = 'B'
                    elif j == len(syn_wd)-1:
                        cws = 'E'
                    else:
                        cws = 'M'
                    new_insts.append(Instance(swd, cws, 'O'))
            else:
                new_insts.extend(ins)
    return new_insts


def synonyms_replace_bi(insts):
    seg_insts = []
    one_wd = []
    is_start = False
    for i, inst in enumerate(insts):
        one_wd.append(inst)
        if inst.cws.lower() == 'b':
            is_start = True
            if i == len(insts)-1 or insts[i+1].cws.lower() != 'i':
                seg_insts.append(one_wd)
                one_wd = []
                is_start = False
        elif inst.cws.lower() == 'i' and (i == len(insts)-1 or insts[i+1].cws.lower() != 'i'):
            if is_start:
                seg_insts.append(one_wd)
                is_start = False
            one_wd = []

    new_insts = []
    for ins in seg_insts:
        if ins[0].ner != 'O':
            new_insts.extend(ins)
        else:
            wds, _ = synonyms.nearby(''.join([s.wd for s in ins]))
            if wds and random.random() < 0.5:
                syn_wd = random.choice(wds[1:6])
                for j, swd in enumerate(syn_wd):
                    if len(syn_wd) == 1:
                        cws = 'B'
                    elif j == 0:
                        cws = 'B'
                    else:
                        cws = 'I'
                    new_insts.append(Instance(swd, cws, 'O'))
            else:
                new_insts.extend(ins)
    return new_insts


def synonyms_replace(insts, rate=0.5):
    for i, inst in enumerate(insts):
        if inst.ner.upper() == 'O':
            wds = synonyms.nearby(inst.wd)
            if wds is not None and random.random() < rate:
                syn_wd = random.choice(wds)
                insts[i].wd = syn_wd
    return insts
        

def replace_ner(insts, ner_dic, rate=0.5):
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

    has_ent = False
    for i, items in enumerate(ners):
        if items[0].ner != 'O' and random.random() < rate:
            has_ent = True
            ner_type = items[0].ner.split('-')[1]
            cands = ner_dic[ner_type]
            new_wd = random.sample(cands, 1)[0]
            new_wds = new_wd.split('_') 
            rep_insts = []
            for j, w in enumerate(new_wds):
                if len(new_wds) == 1:
                    ner_tag = 'S-' + ner_type
                elif j == 0:
                    ner_tag = 'B-' + ner_type
                elif j == len(new_wds) - 1:
                    ner_tag = 'E-' + ner_type
                else:
                    ner_tag = 'I-' + ner_type
                rep_insts.append(Instance(w, ner_tag))
            ners[i] = rep_insts

    return list(itertools.chain.from_iterable(ners))  # flat list
    #if has_ent:
    #    return list(itertools.chain.from_iterable(ners))  # flat list
    #else:
    #    return []


# 对数据中同类型实体进行替换
def aug_ner(data_path, save_path, ner_path, rate=0.5, aug_times=1, seed=1347):
    ner_dic = load_dic(ner_path)
    #dataset = read_data(data_path, ratio, seed)
    dataset = read_data(data_path, 1, seed)
    print(len(dataset))

    repl_insts = []
    for _ in range(aug_times):
        for i, sent in enumerate(dataset):
            sent_insts = replace_ner(sent, ner_dic, rate) 
            sent_insts = synonyms_replace(sent_insts, 1-rate)
            #if sent_insts and sent_insts not in repl_insts:
            if sent_insts:
                repl_insts.append(sent_insts)

    my_insts = repl_insts if repl_insts else dataset

    # for sent_insts in my_insts:
    #     for wd_insts in sent_insts:
    #         for ins in wd_insts:
    #             fw.write(f'{ins.wd}\t{ins.cws}\t{ins.ner}\n')
    #     fw.write('\n')

    print('writing ....')    
    fw = open(save_path, 'w', encoding='utf-8')
    for insts in my_insts:
        for ins in insts:
            fw.write(f'{ins.wd}\t{ins.ner}\n')
        fw.write('\n')

    fw.close()
    print('Done!')


# python aug_utils.py onto4.train.1415 train.1415.repl1 train.dic 1 1
def run():
    train_file, out_file, train_dic_file, times, rate = sys.argv[1:6]
    print('times:',times)
    aug_ner(train_file,  # 训练集路径
            out_file,    # 输出文件
            train_dic_file,  # 训练集实体词典
            aug_times=int(times),  # 0表示不替换
            rate=float(rate))


def run_split():
    random.seed(2379)
    train_file, ratio = sys.argv[1:3]
    _, part_set = rand_train_dev_split(train_file, float(ratio), 2379)
    random.shuffle(part_set)
    dev_size = int(len(part_set) * 0.1)
    write_to_conll('conll.train.'+str(len(part_set)-dev_size), part_set[dev_size:])    
    write_to_conll('conll.dev.'+str(len(part_set)-dev_size), part_set[:dev_size])    


run()
#run_split()

