
def reader(path):
    inst = {'wds': [], 'ner_tags': []}
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if line.strip() == '' or len(tokens) < 2:
                if len(inst['wds']) > 0:
                    yield inst
                inst = {'wds': [], 'ner_tags': []}
            elif len(tokens) == 2:
                inst['wds'].append(tokens[0])
                inst['ner_tags'].append(tokens[-1])

    if len(inst['wds']) > 0:
        yield inst


def bio2bioes(inst):
    ners = inst['ner_tags']
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


def count(in_path):
    N = 0
    for inst in reader(in_path):
        N += 1
    print(N)


def read_conll(in_path, out_path):
    fw = open(out_path, 'w', encoding='utf-8')
    for inst in reader(in_path):
        bio2bioes(inst)
        for wd, ner in zip(inst['wds'], inst['ner_tags']):
            fw.write(f'{wd}\t{ner}\n')
        fw.write('\n')
    fw.close()
    print('Done!')


def test_1():
    count('conll.train.675')
    count('conll.dev.675')


def filter():
   for ci, ai in zip(reader('conll.train.675'), reader('conll.train.675.repl2')):
       if len(ci['wds']) >= 10 and len(ci['wds']) <= 15:
           print(ci['wds'], '\n', ai['wds'], '\n\n')
      

filter()


