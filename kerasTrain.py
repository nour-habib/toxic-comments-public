import sys, os, re, csv, codecs, optparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import string
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
import kerasModels

path = './data/input/'
EMBEDDING_FILE = f'{path}embeddings/glove.6B.50d.txt'
TRAIN_FILE = f'{path}train_preprocessed.csv'
TEST_FILE = f'{path}test_preprocessed.csv'
embed_size = 50
max_features = 20000
maxlen = 200

def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def generateCSVName(modelType, useGloVE, bidirectional):
    csv_name = modelType
    if bidirectional == True:
        csv_name = csv_name + "-bidirectional"
    if useGloVE == True:
        csv_name = csv_name + "-GloVE"
    return csv_name

def runClassifier(modelType, useGloVE, bidirectional):
    train = pd.read_csv(TRAIN_FILE, keep_default_na=False)
    #keep_default_na parameter makes pandas read empty cells as NULL instead of float("NaN")
    test = pd.read_csv(TEST_FILE, keep_default_na=False)

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]

    # TOKENIZER IS BUILT IN WOWZA WAY TO MAKE ME WASTE A WHOLE DAY PYTORCH
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    # Pad sequence
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

    if useGloVE == True:
        # Read glove word vectors into dict word->vector
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

        # Random initializations for words that aren't in GloVE
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        print(f"Embedding mean and std: [{emb_mean}, {emb_std}]")
        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))

        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix = None

    # Build model
    if modelType == "LSTM" or modelType == "RNN" or modelType == "GRU":
        model = kerasModels.RNNBaselineModels(maxlen, max_features, embed_size, model_name=modelType, bidirectional=bidirectional, embedding_matrix=embedding_matrix)
    elif modelType == "CNN":
        model = kerasModels.CNNBaseline(maxlen, max_features, embed_size, embedding_matrix)
    elif modelType == "FC":
        model = kerasModels.FullyConnectedBaseline(maxlen, max_features, embed_size, embedding_matrix)

    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.Adam(learning_rate =0.01),
                    metrics=['accuracy'])

    batch_size = 32
    epochs = 10
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping]

    model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

    y_test = model.predict(X_te)

    # Make predictions
    submission_df = pd.DataFrame(columns=['id'] + list_classes)
    submission_df['id'] = test['id'].values 
    submission_df[list_classes] = y_test 

    # Generate name of csv file
    csv_name = generateCSVName(modelType, useGloVE, bidirectional)
    submission_df.to_csv(f"./data/output/{csv_name}.csv", index=False)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-e", "--embedding", dest="useGloVE", default=False, action= "store_true", help="Toggle the GloVE embeddings for training")
    optparser.add_option("-m", "--model", dest="modelType", default="LSTM", help="Choose type of model to use: [LSTM, RNN, GRU, CNN, FC]")
    optparser.add_option("-b", "--bidirectional", dest="bidirectional", default=False, action = "store_true", help="Toggle if RNN-based models are bidirectional")
    (opts, _) = optparser.parse_args()

    modelType = opts.modelType
    useGloVE = opts.useGloVE
    bidirectional = opts.bidirectional
    
    # Only allow bidirectional if the model is RNN based
    if bidirectional and (modelType == "CNN" or modelType == "FC"):
        bidirectional = False

    runClassifier(modelType, useGloVE, bidirectional)