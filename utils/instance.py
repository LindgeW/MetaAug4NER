class Instance(object):
    def __init__(self, chars, ner_tags, **kwargs):
        self.chars = chars
        self.ner_tags = ner_tags
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


