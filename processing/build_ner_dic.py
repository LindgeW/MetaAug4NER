import sys

# extract entities from CoNLL dataset


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


def extract_ners(inst):
    ners = []
    one_ent = []
    start = False
    for i, (wd, ner) in enumerate(zip(inst['wd'], inst['ner'])):
        if ner == 'O':
            start = False
            one_ent = []
            continue
        elif ner.startswith('S-'):
            ners.append(([wd], ner.split('-')[1]))
            start = False
            one_ent = []
        elif ner.startswith('B-'):
            one_ent = [wd]
            start = True
        elif ner.startswith('E-'):
            if start and one_ent:
                one_ent.append(wd)
                ners.append((one_ent, ner.split('-')[1]))
                one_ent = []
                start = False
        else:
            if ner.startswith('M-') or ner.startswith('I-'):
                one_ent.append(wd)
    return ners


def build_ner_dic(in_path, out_path, lang='cn'):
    assert lang in ['en', 'cn']
    fw = open(out_path, 'w', encoding='utf-8')
    ent_dic = dict()
    for inst in load_data(in_path):
        for wds, tp in extract_ners(inst):
            sep = '' if lang == 'cn' else chr(1)
            ent = sep.join(wds)
            if len(ent_dic) > 0 and tp in ent_dic:
                ent_dic[tp].add(ent)
            else:
                ent_dic[tp] = {ent}

    for tp, wds in ent_dic.items():
        fw.write(f'{tp} ||| {" ".join(wds)}\n')
    fw.close()
    print('Done!')


def run():
    train_file, dic_file, lang = sys.argv[1:4]
    build_ner_dic(train_file, dic_file, lang)


run()


