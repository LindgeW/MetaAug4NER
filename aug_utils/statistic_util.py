import sys
import os
import glob


class Instance:
    def __init__(self, wd, cws, ner):
        self.wd = wd
        self.cws = cws
        self.ner = ner

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

def read_file(path):
    assert os.path.exists(path)
    sents = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < 3:
                if len(sents) > 0:
                    yield sents
                sents = []
            else:
                sents.append(Instance(*tokens))
    if len(sents) > 0:
        yield sents


def discard_ner_types(file_path):
    out_path = file_path + '_3'
    fw = open(out_path, 'w', encoding='utf-8')
    for insts in read_file(file_path):
        for inst in insts:
            if inst.ner.startswith('B-') or inst.ner.startswith('I-'):
                ner_type = inst.ner.split('-')[1]
                if ner_type not in ['ORG', 'PERSON', 'LOC']:
                    inst.ner = 'O'
            fw.write(f'{inst.wd}\t{inst.cws}\t{inst.ner}\n')
        fw.write('\n')
    fw.close()


from collections import Counter
def sent_count(file_path):
    num_sent = 0
    num_wd = 0
    ent_set = Counter()
    for insts in read_file(file_path):
        num_sent += 1
        num_wd += len(insts)
        for inst in insts:
            if inst.ner != 'O':
                bound, ent_type = inst.ner.split('-')
                if bound == 'B':
                    ent_set[ent_type] += 1
    print(f'#sent: {num_sent}, #wd: {num_wd}, #ent: {len(ent_set)}, ents: {ent_set}')


def compare(file1, file2):
    insts1 = []
    insts2 = []
    for insts in read_file(file1):
        insts1.extend(insts)

    for insts in read_file(file2):
        insts2.extend(insts)

    print(len(insts1), len(insts2))
    same_insts = set(insts1) & set(insts2)
    print(same_insts, len(same_insts))


def run2():
    in_path, out_path = sys.argv[1:3]
    #discard_ner_types(in_path)
    sent_count(in_path)    
    #compare(in_path, out_path)
    print('finished')

run2()

