from gensim.models import Word2Vec, KeyedVectors
wd_vec_path = 'glove.6B.100d.txt'

stop_wds = ['[',']', '(', ')', '.',  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'other', 'such', 'no', 'nor', 'not', 'only', 'own', 'so', 'than', 's', 't', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

class Synonym:
    def __init__(self, path=wd_vec_path):
        #wv_from_text.save_word2vec_format('glove.6b.200d.bin', binary=True)
        #self.wv_from_text = KeyedVectors.load_word2vec_format(wd_vec_path, binary=True)
        self.wv_from_text = KeyedVectors.load_word2vec_format(wd_vec_path, binary=False)

    def nearby(self, wd, topk=5):
        try:
            if wd in stop_wds:
                return None

            sims = self.wv_from_text.most_similar(wd.lower(), topn=topk)
        except Exception as e:
            return None

        return [w for w, s in sims]


#synonyms = Synonym()
#res = synonyms.nearby('sad')
#print(res)
