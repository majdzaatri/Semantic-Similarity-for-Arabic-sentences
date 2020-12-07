
class Vectorizer:

    def __init__(self, word_embedding_model):
        self.word_embedding_model = word_embedding_model
        # self.tokenizer = tokenizer

    def vectorize_sentence(self, sentence, threshold = -1):
                                                                            # nltk.download('punkt')
                                                                            # tokens = word_tokenize(sentence)
                                                                            # # tokens = self.tokenizer.tokenize(sentence)
                                                                            # if threshold > 0:
                                                                            #     tokens = tokens[:threshold]
                                                                            #

                                                                            # for token in tokens:
                                                                            #     token_vector = self.word_embedding_model.get_vector(token)
                                                                            #     if token_vector is not "unknown":
                                                                            #         vector.append(token_vector)

        vector = []

        for word in sentence:
            word_vector = self.word_embedding_model.get_vector(word)
            if word_vector is not "unknown":
                vector.append(word_vector)

        return vector


    def vectorize_sentences(self, df, sentences, threshold = -1):
        return [self.vectorize_sentence(s) for s in sentences]

    def vectorize_df(self, df):
        a_vectors = [self.vectorize_sentence(sentence) for sentence in df['book1']]
        import math
        import pandas as pd
        for sentence in df['book2']:
            if sentence != 'nan':
                print(sentence)

        b_vectors = [self.vectorize_sentence(sentence) for sentence in df['book2']]

        return a_vectors, b_vectors