class Instance(object):
    def __init__(self, chars: list, ner_tags: list):
        self.chars = chars
        self.ner_tags = ner_tags

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)



