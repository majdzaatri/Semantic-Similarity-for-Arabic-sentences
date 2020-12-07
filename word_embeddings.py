

class WordEmbeddings:

    def __init__(self, word_embedding_model):
        self.word_embedding_model = word_embedding_model

    def get_vector(self, word):

        if any(char.isdigit() for char in word):
            return "unknown"
        elif word in self.word_embedding_model:
            return self.word_embedding_model.wv[word]
        else:
            return "unknown"