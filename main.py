
from preprocessing import *
from utilities import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


if __name__ == '__main__':

    concatenate_files('books/arab/t1/*.txt','bookt1')
    f = open('bookt1', encoding='urf-8')
    book = f.read()

    text = remove_punctuations(book)
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_repeating_char(text)

    # word_embedding_model = gensim.models.Word2Vec.load('models/wikipedia_sg_300')
    word_embedding_model, _, max_features, embed_size = get_init_parameters('models/full_grams_cbow_100_twitter.mdl')

    # find and print the most similar terms to a word
    most_similar = word_embedding_model.wv.most_similar(text.split()[0])


    for term, score in most_similar:
        print(term, score)

    # get a word vector
    splited_text = text.split()[0:201]
    word_vector = []
    for word in splited_text:
        try:
            word_vector.append(word_embedding_model.wv[word])
        except:
            print(word,"word not in corpus")

    inp = Input(shape=())
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=(300, 1)),
        Bidirectional(LSTM(100, return_sequences=True), input_shape=(300, 1))
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.fit(words_vector[0:200], epochs=3, batch_size=64, steps_per_epoch=1000)
    # score = model.evaluate(words_vector[200:400])
    # print(score)
