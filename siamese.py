# from tensorflow.keras.layers import *
# from keras_self_attention import SeqSelfAttention
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import Model
from attention import Attention
import tensorflow as tf


# from attention_layer import AttentionLayer

class SiameseModel:

    def __init__(self):
        n_hidden = 100
        input_shape = (126, 100)

        Bilstm_1 = LSTM(n_hidden, return_sequences=True, input_shape=input_shape)
        Bilstm_2 = LSTM(n_hidden, return_sequences=True, input_shape=input_shape)

        self_attention = SeqSelfAttention(attention_activation='sigmoid')
        attention = Attention()
        non_linear = Activation('relu')

        left_input = Input(shape=input_shape, name='input_1')
        right_input = Input(shape=input_shape, name='input_2')

        first_left_lstm_output = Bilstm_1(left_input)
        left_self_attention_output = self_attention(first_left_lstm_output)
        second_left_lstm_output = Bilstm_2(first_left_lstm_output)

        first_right_lstm_output = Bilstm_2(right_input)
        right_self_attention_output = self_attention(first_right_lstm_output)
        second_right_lstm_output = Bilstm_2(first_right_lstm_output)

        left_attention_output = tf.keras.layers.Attention()([second_left_lstm_output, right_self_attention_output])
        right_attention_output = tf.keras.layers.Attention()([second_right_lstm_output, left_self_attention_output])

        left_concatenation = left_self_attention_output + left_attention_output
        right_concatenation = right_self_attention_output + right_attention_output

        left_nonlinear_output = non_linear(left_concatenation)
        right_nonlinear_output = non_linear(right_concatenation)

        mlp_input = Concatenate([left_nonlinear_output, right_nonlinear_output])

        self.model = Model([Bilstm_1, Bilstm_2])


s = SiameseModel()
