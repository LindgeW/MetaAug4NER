'''
CWS / POS标签规范：
1、BI
2、BIS
3、BIES

NER标签规范：
1、BIO
2、BISO
3、BIESO
'''


def bi2bies(bi_tags):
    tag_len = len(bi_tags)
    for i, t in enumerate(bi_tags):
        if t == 'B':
            if i + 1 == tag_len or 'I' != bi_tags[i+1]:
                bi_tags[i] = 'S'
        elif t == 'I':
            if i + 1 == tag_len or 'I' != bi_tags[i+1]:
                bi_tags[i] = 'E'
    return bi_tags


# BIO -> BIOES
def bio2bioes(bio_tags):
    tag_len = len(bio_tags)
    for i, t in enumerate(bio_tags):
        if 'B-' in t and (i+1 == tag_len or 'I-' not in bio_tags[i+1]):
            _type = bio_tags[i].split('-')[1]
            bio_tags[i] = 'S-' + _type
        elif 'I-' in t and (i+1 == tag_len or 'I-' not in bio_tags[i+1]):
            _type = bio_tags[i].split('-')[1]
            bio_tags[i] = 'E-' + _type
    return bio_tags


# =============================CWS============================== #


# BI
def extract_cws_bi_span(tag_seq):
    spans = []
    s = 0
    n = len(tag_seq)
    start = False
    for i, tag in enumerate(tag_seq):
        if tag == 'B':
            s = i
            if i + 1 == n or tag_seq[i + 1] != 'I':
                spans.append((s, i))
                start = False
            else:
                start = True
        elif tag == 'I':
            if start:
                if i + 1 == n or tag_seq[i + 1] != 'I':
                    spans.append((s, i))
                    start = False
        else:
            start = False
    return spans


# BIS
def extract_cws_bis_span(tag_seq):
    spans = []
    s = 0
    n = len(tag_seq)
    start = False
    for i, tag in enumerate(tag_seq):
        if tag == 'S':
            spans.append((i, i))
            start = False
        elif tag == 'B':
            s = i
            start = True
        elif tag == 'I':
            if start:
                if i + 1 == n or tag_seq[i + 1] != 'I':
                    spans.append((s, i))
                    start = False
        else:
            start = False

    return spans


# BIES / BMES
def extract_cws_bies_span(tag_seq):
    spans = []
    s = 0
    start = False
    for i, tag in enumerate(tag_seq):
        if tag == 'S':
            spans.append((i, i))
            start = False
        elif tag == 'B':
            s = i
            start = True
        elif tag == 'E':
            if start:
                spans.append((s, i))
                start = False
        else:
            if tag not in ['I', 'M']:
                start = False
    return spans


# =============================NER============================== #


def extract_ner_bio_span(tag_seq: list):
    span_res = []
    n = len(tag_seq)
    s = 0
    type_b = None
    start = False
    for i, tag in enumerate(tag_seq):
        if tag == 'O':
            start = False
        elif tag.startswith('B-'):
            s = i
            type_b = tag.split('-')[1]
            if i + 1 == n or not tag_seq[i+1].startswith('I-'):
                span_res.append((s, i, type_b))
                start = False
            else:
                start = True
        elif tag.startswith('I-'):
            if start and tag.split('-')[1] == type_b:
                if i + 1 == n or not tag_seq[i+1].startswith('I-'):
                    span_res.append((s, i, type_b))
                    start = False
    return span_res


def extract_ner_biso_span(tag_seq):
    spans = []
    s = 0
    n = len(tag_seq)
    start = False
    type_b = None
    for i, tag in enumerate(tag_seq):
        if tag == 'O':
            start = False
        elif tag.startswith('S-'):
            spans.append((i, i, tag.split('-')[1]))
            start = False
        elif tag.startswith('B-'):
            s = i
            start = True
            type_b = tag.split('-')[1]
        elif tag.startswith('I-'):
            if start and type_b == tag.split('-')[1]:
                if i + 1 == n or not tag_seq[i+1].startswith('I-'):
                    spans.append((s, i, type_b))
                    start = False
    return spans


def extract_ner_bieso_span(tag_seq):
    spans = []
    s = 0
    start = False
    type_b = None
    for i, tag in enumerate(tag_seq):
        if tag == 'O':
            start = False
        elif tag.startswith('S-'):
            spans.append((i, i, tag.split('-')[1]))
            start = False
        elif tag.startswith('B-'):
            s = i
            type_b = tag.split('-')[1]
            start = True
        elif tag.startswith('E-'):
            if start and tag.split('-')[1] == type_b:
                spans.append((s, i, type_b))
                start = False
        else:
            if '-' not in tag or tag.split('-')[1] != type_b:
                start = False

    return spans


def seq_match(main_str, sub_strs: list):
    N = len(main_str)
    match_spans = set()
    for sub_str in sub_strs:
        L = len(sub_str)
        for i in range(N - L + 1):
            if main_str[i: i+L] == sub_str:
                match_spans.add((i, i+L-1))
    return match_spans


def span2tags(spans, seq_len, default_tag='O'):
    '''
    :param spans: [(s, e, cls), ...]
    :param seq_len: sequence length
    :param default_tag: default tag in sequence
    :return:
    '''
    tags = [default_tag] * seq_len
    for one_span in spans:
        if len(one_span) == 3:
            s, e, cls = one_span
            cls = '-'+cls
        elif len(one_span) == 2:
            s, e = one_span
            cls = ''
        else:
            raise ValueError

        tags[s] = 'B' + cls
        tags[e] = 'E' + cls
        if s == e:
            tags[s] = 'S' + cls
        elif s < e:
            tags[s+1: e] = ['I' + cls] * (e - s - 1)
        else:
            raise IndexError
    return tags


def test_():
    # x = 'I I S B S I B X I S B I I S S B I S S B I I'.split()
    # x = 'I I B B X I B B I I B I B I B B I B I B I I'.split()
    # x = 'B I E S B E S S B I I E B I S B E S E'.split()
    # y = extract_cws_bi_span(x)
    # y = extract_cws_bis_span(x)
    # y = extract_cws_bies_span(x)

    # x = 'O I-per S-org S-org I-per S-org B-loc I-loc E-loc O B-org E-org O B-per E-per B-loc E-per O S-LOC E-LOC'.split()
    x = 'O B-PER I-PER I-LOC B-PER I-PER B-LOC B-ORG I-ORG O B-LOC I-LOC I-PER'.split()
    y = extract_ner_bio_span(x)
    # y = extract_ner_biso_span(x)
    # y = extract_ner_bieso_span(x)
    print(y)

    spans = [(1, 2), (3, 3), (4, 6)]
    tags = span2tags(spans, 10)
    print(tags)

    main_str = '我是一个目前在阿里巴巴实习的研究生，方向是NLP'
    sub_str = ['阿里巴巴', 'NLP']
    res = seq_match(main_str, sub_str)
    tags = span2tags(res, len(main_str))
    print(res)
    print(tags)
