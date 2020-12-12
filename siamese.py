from tensorflow.keras.layers import *
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Model
class SiameseModel:

    def __init__(self):
        n_hidden = 100
        input_shape = (126, 100)

        Bilstm_1 = Bidirectional(LSTM(n_hidden, return_sequences=True, input_shape= input_shape))
        Bilstm_2 = Bidirectional(LSTM(n_hidden))
        self_attention = SeqSelfAttention(attention_activation='sigmoid')
        attention = Attention()
        non_linear = Activation('relu')

        left_input = Input(shape=(100,), name='input_1')
        right_input = Input(shape=(100,), name='input_1')


        first_left_lstm_output = Bilstm_1(left_input)
        left_self_attention_output = self_attention(first_left_lstm_output)
        second_left_lstm_output = Bilstm_2(first_left_lstm_output)

        first_right_lstm_output = Bilstm_2(right_input)
        right_self_attention_output = self_attention(first_left_lstm_output)
        second_right_lstm_output = Bilstm_2(first_right_lstm_output)


        left_attention_output = attention([second_left_lstm_output,right_self_attention_output])
        right_attention_output = attention([second_right_lstm_output, left_self_attention_output])

        left_concatenation = Concatenate([left_self_attention_output,left_attention_output])
        right_concatenation = Concatenate([right_self_attention_output, right_attention_output])


        left_nonlinear_output = non_linear(left_concatenation)
        right_nonlinear_output = non_linear(right_concatenation)

        mlp_input = Concatenate([left_nonlinear_output,right_nonlinear_output])

        self.model = Model([Bilstm_1, Bilstm_2])

