import os
import gensim
import glob
import shutil
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_init_parameters(path):
    word_embedding_model = gensim.models.Word2Vec.load(path)
    n_words = len(word_embedding_model.wv.vocab)
    vocab_dim = word_embedding_model.wv[word_embedding_model.wv.index2word[0]].shape[0]
    index_dict = dict()
    for i in range(n_words):
        index_dict[word_embedding_model.wv.index2word[i]] = i + 1
    return word_embedding_model, index_dict, n_words, vocab_dim



# def split_datasets(book, test_size, seed=42):
def split_datasets():
    x = []
    book = open("text.txt", encoding="utf-8")
    for line in book:
        temp = line.split(',')
        x.append(temp[0])
        print(x)
    print("new x")



def concatenate_files(path, new_file_title):
    with open(new_file_title, 'wb') as outfile:
        for filename in glob.glob(path):
            if filename == 'concat':
                # don't want to copy the output into the output
                continue
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)


def get_DataFrame(book1, col_name):

    n = 20
    i = 2
    words1 = iter(book1.split())
    lines1, current1 = [], next(words1)

    for word in words1:
        if i > n:
            lines1.append(current1)
            current1 = word
            i = 2
        else:
            current1 += " " + word
            i+=1

    lines1.append(current1)

    import pandas as pd
    df1 = pd.DataFrame(lines1, columns=[col_name])
    return df1

def pad_tensor(tensor, max_len, dtype='float32'):
    return pad_sequences(tensor, padding='post', dtype=dtype, maxlen=max_len)