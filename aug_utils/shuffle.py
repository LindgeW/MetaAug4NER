import numpy as np
import sys
import os


class Instance:
    def __init__(self, wd, cws, ner):
        self.wd = wd
        self.cws = cws
        self.ner = ner

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


def read_from_file(path):
    assert os.path.exists(path)
    sents = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split('\t')
            if line.strip() == '' or len(tokens) < 3:
                if len(sents) > 0:
                    yield sents
                sents = []
            else:
                sents.append(Instance(*tokens))
    if len(sents) > 0:
        yield sents


def shuffle(in_path, out_path):
    loader = read_from_file(in_path)
    data_set = []
    for insts in loader:
        data_set.append(insts)

    np.random.shuffle(data_set)

    with open(out_path, 'w', encoding='utf-8') as fw:
        for insts in data_set:
            for inst in insts:
                fw.write(f'{inst.wd}\t{inst.cws}\t{inst.ner}\n')
            fw.write('\n')
    print('Done')


def run():
    train_file, out_file = sys.argv[1:3]
    shuffle(train_file, out_file)

run()
