# Keras Models

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SimpleRNN, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten, Conv1D, MaxPooling1D
from keras.models import Model
def RNNBaselineModels(maxlen, max_features, embed_size, model_name='LSTM', bidirectional=False, embedding_matrix = None):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is not None:
        print("Using GLoVE embeds")
        x = Embedding(max_features, embed_size, weights = [embedding_matrix])(inp)
    else:
        print("Not using GLoVE embeds")
        x = Embedding(max_features, embed_size)(inp)

    if model_name == 'RNN':
        if bidirectional:
            print("Running bidirectional RNN")
            #redacted
        else:
            print("Running RNN")
            #redacted
    elif model_name == 'GRU':
        if bidirectional:
            print("Running bidirectional GRU")
            #redacted
        else:
            print("Running GRU")
            #redacted
    else:
        # Default to LSTM
        if bidirectional:
            print("Running bidirectional LSTM")
            #redacted
        else:
            print("Running LSTM")
           #redacted

    #redacted

    model = Model(inputs=inp, outputs = x)
    return model

def FullyConnectedBaseline(maxlen, max_features, embed_size, embedding_matrix = None):
    #redacted

def CNNBaseline(maxlen, max_features, embed_size, embedding_matrix = None):
    #redacted
