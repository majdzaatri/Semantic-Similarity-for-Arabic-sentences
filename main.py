
from preprocessing import *
from utilities import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import tensorflow as tf
import pandas as pd
from pathlib import Path

from word_embeddings import WordEmbeddings
from tokenizer import Tokenizer
from vectorizer import  Vectorizer
from siamese import SiameseModel

if __name__ == '__main__':

    concatenate_files('books/arab/t1/*.txt','bookt1')
    f = open('bookt1', encoding='utf-8')
    book1 = f.read()

    concatenate_files('books/arab/t2/*.txt', 'bookt2')
    f = open('bookt2', encoding='utf-8')
    book2 = f.read()

    book1,book2 = remove_punctuations(book1),remove_punctuations(book2)
    # book1,book2 = normalize_arabic(book1), normalize_arabic(book2)
    book1,book2 = remove_diacritics(book1), remove_diacritics(book2)
    book1,book2 = remove_repeating_char(book1), remove_repeating_char(book2)

    df1 = get_DataFrame(book1,'book1')
    df2 = get_DataFrame(book2,'book2')

    df = pd.concat([df1,df2],names = ["book1","book2"], ignore_index=False, axis=1)
    df.to_csv("list1.csv", index=False)

    # word_embedding_model = gensim.models.Word2Vec.load('models/wikipedia_sg_300')
    word_embedding_model, _, max_features, embed_size = get_init_parameters('models/full_grams_cbow_100_twitter.mdl')
    word_embedding = WordEmbeddings(word_embedding_model)
    tokenizer = Tokenizer()
    vectorizer = Vectorizer(word_embedding)

    df.dropna(subset=["book2"], inplace=True)

    train_a_vectors, train_b_vectors = vectorizer.vectorize_df(df)
    train_max_a_length = len(max(train_a_vectors, key=len))
    train_max_b_length = len(max(train_b_vectors, key=len))
    print("max number of tokens per sentence A is %d" % train_max_a_length)
    print("max number of tokens per sentence B is %d" % train_max_b_length)
    max_len = max([train_max_a_length,train_max_b_length])

    train_a_vectors = pad_tensor(train_a_vectors, max_len)
    train_b_vectors = pad_tensor(train_b_vectors, max_len)

    siameseModel = SiameseModel()
    siameseModel.summary()
    # model = Sequential()
    # model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(126, 100))))
    # model.add(Bidirectional(LSTM(100)))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    #
    # labels = [1]*5811
    #
    # model.fit(train_a_vectors, labels, epochs=100,  batch_size=32)
    #

    #
    # # model.fit(words_vector[0:200], epochs=3, batch_size=64, steps_per_epoch=1000)
    # # score = model.evaluate(words_vector[200:400])
    # # print(score)
