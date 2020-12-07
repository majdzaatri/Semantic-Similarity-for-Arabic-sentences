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

    n = 250

    words1 = iter(book1.split())
    # words2 = iter(book2.split())

    lines1, current1 = [], next(words1)
    # lines2, current2 =[], next(words2)

    for word in words1:
        if len(current1) + 1 + len(word) > n:
            lines1.append(current1)
            current1 = word
        else:
            current1 += " " + word

    # for word in words2:
    #     if len(current2) + 1 + len(word) > n:
    #         lines2.append(current2)
    #         current2 = word
    #     else:
    #         current2 += " " + word
    #
    lines1.append(current1)
    # lines2.append(current2)

    import pandas as pd
    df1 = pd.DataFrame(lines1, columns=[col_name])
    # df2 = pd.DataFrame(lines2, columns=['book2'])
    # df = pd.concat([df1,df2])
    return df1

def pad_tensor(tensor, max_len, dtype='float32'):
    return pad_sequences(tensor, padding='post', dtype=dtype, maxlen=max_len)