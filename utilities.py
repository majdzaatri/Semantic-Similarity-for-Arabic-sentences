import os
import gensim
import glob
import shutil

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
