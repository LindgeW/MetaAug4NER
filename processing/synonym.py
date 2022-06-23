from gensim.models import Word2Vec, KeyedVectors
from stop_words import stop_words

# Online synonym toolkit：Wikipedia2Vec
# https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
# https://wikipedia2vec.github.io/wikipedia2vec/usage/

w2v_path = './glove.6B.100d.txt'

class Synonym:
    def __init__(self, word2vec_path):
        # wv_from_text.save_word2vec_format('glove.6b.200d.bin', binary=True)
        # self.wv_from_text = KeyedVectors.load_word2vec_format(wd_vec_path, binary=True)
        self.wv_from_text = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)

    def nearby(self, wd, topk=5):
        try:
            if wd in stop_words:
                return None
            sims = self.wv_from_text.most_similar(wd.lower(), topn=topk)
        except Exception as e:
            return None

        return [w for w, s in sims]


synonyms = Synonym(w2v_path)