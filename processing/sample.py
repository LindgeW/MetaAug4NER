import sys
import random


def load_data(path):
    inst = {'wd': [], 'ner': []}
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < 2:
                if len(inst['wd']) > 0:
                    yield inst
                inst = {'wd': [], 'ner': []}
            else:
                inst['wd'].append(tokens[0])
                inst['ner'].append(tokens[-1])
    if len(inst['wd']) > 0:
        yield inst


def write_to_conll(path, insts):
    nb_sent = 0
    with open(path, 'w', encoding='utf-8') as fw:
        for inst in insts:
            nb_sent += 1
            for wd, ner in zip(inst['wd'], inst['ner']):
                fw.write(f'{wd}\t{ner}\n')
            fw.write('\n')
    print(nb_sent, ' sents are written to ', path)


def sampleN(path, num=500, seed=None):
    if seed:
        random.seed(seed)
    dataset = []
    for insts in load_data(path):
        dataset.append(insts)
        if len(dataset) == num:
            break
    random.shuffle(dataset)
    return dataset


def train_dev_split(path, ratio=0.1, seed=None):
    if seed:
        random.seed(seed)
    dataset = []
    for insts in load_data(path):
        dataset.append(insts)
    random.shuffle(dataset)
    n_split = int(len(dataset) * ratio)
    train_set, dev_set = dataset[n_split:], dataset[:n_split]
    print(len(dataset), len(train_set), len(dev_set))
    return train_set, dev_set


def run_split():
    seed = 1234
    random.seed(seed)
    train_file, ratio = sys.argv[1:3]
    part_set = train_dev_split(train_file, float(ratio), seed)[1]
    random.shuffle(part_set)
    n = len(part_set)
    dev_size = int(n * 0.1)
    write_to_conll('train.' + str(n - dev_size), part_set[dev_size:])
    write_to_conll('dev.' + str(n - dev_size), part_set[:dev_size])
    print('Done!')


run_split()