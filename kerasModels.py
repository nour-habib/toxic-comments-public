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
            x = Bidirectional(SimpleRNN(60, return_sequences = True, name='rnn_layer'))(x)
        else:
            print("Running RNN")
            x = SimpleRNN(60, return_sequences = True, name='rnn_layer')(x)
    elif model_name == 'GRU':
        if bidirectional:
            print("Running bidirectional GRU")
            x = Bidirectional(GRU(60, return_sequences = True, name='gru_layer'))(x)
        else:
            print("Running GRU")
            x = GRU(60, return_sequences = True, name='gru_layer')(x)
    else:
        # Default to LSTM
        if bidirectional:
            print("Running bidirectional LSTM")
            x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer'))(x)
        else:
            print("Running LSTM")
            x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs = x)
    return model

def FullyConnectedBaseline(maxlen, max_features, embed_size, embedding_matrix = None):
    inp = Input(shape = (maxlen,))
    if embedding_matrix is not None:
        print("Using GLoVE embeds")
        x = Embedding(max_features, embed_size, weights = [embedding_matrix])(inp)
    else:
        print("Not using GLoVE embeds")
        x = Embedding(max_features, embed_size)(inp)
    print("Running FC")
    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dense(16, activation = 'relu')(x)
    x = Dense(6, activation = 'sigmoid')(x)

    FCmodel = Model(inputs=inp, outputs=x)
    return FCmodel

def CNNBaseline(maxlen, max_features, embed_size, embedding_matrix = None):
    inp = Input(shape = (maxlen,))
    if embedding_matrix is not None:
        print("Using GLoVE embeds")
        x = Embedding(max_features, embed_size, weights = [embedding_matrix])(inp)
    else:
        print("Not using GLoVE embeds")
        x = Embedding(max_features, embed_size)(inp)
    print("Running CNN")
    x = Conv1D(128, 5, activation = 'relu', padding = 'same')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation = 'relu', padding = 'same')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation = 'relu', padding = 'same')(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(6, activation = 'sigmoid')(x)

    CNNmodel = Model(inputs = inp, outputs = x)
    return CNNmodel 