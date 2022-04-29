import sys
'''
Chinese：char-level CoNLL to word-level CoNLL
'''


def load_data(path):
    inst = {'wd': [], 'cws': [], 'ner': []}
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < 2:
                if len(inst['wd']) > 0:
                    yield inst
                inst = {'wd': [], 'cws': [], 'ner': []}
            else:
                inst['wd'].append(tokens[0])
                inst['ner'].append(tokens[-1])
    if len(inst['wd']) > 0:
        yield inst


def bio2bioes(inst):
    ners = inst['ner']
    n = len(ners)
    for i, ner_tag in enumerate(ners):
        if ner_tag.startswith('B-'):
            if i == n - 1:
                ners[i] = ner_tag.replace('B-', 'S-')
            elif not ners[i+1].startswith('I-'):
                ners[i] = ner_tag.replace('B-', 'S-')
        if ner_tag.startswith('I-'):
            if i == n - 1:
                ners[i] = ner_tag.replace('I-', 'E-')
            elif not ners[i+1].startswith('I-'):
                ners[i] = ner_tag.replace('I-', 'E-')
    return inst


def char2word_bi(inst):
    seg_insts = []
    one_wd = []
    is_start = False
    n = len(inst['cws'])
    for i, cws in enumerate(inst['cws']):
        one_wd.append(inst['wd'][i])
        if cws.lower() == 'b':
            is_start = True
            if i == n - 1 or inst['cws'][i + 1].lower() != 'i':
                seg_insts.append(one_wd)
                one_wd = []
                is_start = False
        elif cws.lower() == 'i' and (i == n - 1 or inst['cws'][i + 1].lower() != 'i'):
            if is_start:
                seg_insts.append(one_wd)
                is_start = False
            one_wd = []
    return seg_insts


def char2word_bies(insts):
    seg_insts = []
    one_wd = []
    is_start = False
    for i, inst in enumerate(insts):
        if inst.cws.lower() not in 'bies':
            is_start = False
            continue
        if inst.cws.lower() == 's':
            seg_insts.append([inst])
            is_start = False
        elif inst.cws.lower() == 'b':
            one_wd = [inst]
            is_start = True
        elif inst.cws.lower() == 'i':
            if is_start:
                one_wd.append(inst)
        elif inst.cws.lower() == 'e':
            if is_start:
                one_wd.append(inst)
                seg_insts.append(one_wd)
                is_start = False
            one_wd = []
    return seg_insts


def ner_type_proc(in_path, out_path):
    '''
    :param in_path:  bio conll
    :param out_path: bioes conll
    :return:
    '''
    with open(out_path, 'w', encoding='utf-8') as fw:
        for inst in load_data(in_path):
            inst = bio2bioes(inst)
            for wd, ner in zip(inst['wd'], inst['ner']):
                fw.write(f'{wd}\t{ner}\n')
            fw.write('\n')
    print('Done!')


def run():
    bio_file, bioes_file = sys.argv[1:3]
    ner_type_proc(bio_file, bioes_file)


run()



