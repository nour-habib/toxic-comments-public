import pandas as pd, csv, os
import re
import string

def preprocess(data):
    to_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    remove_new_line = lambda x: re.sub("\n", " ", x)
    remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
    numletter = lambda x: re.sub('\w*\d\w*', ' ', x)
    data['comment_text'] = data['comment_text'].map(to_lower).map(remove_new_line).map(remove_non_ascii)

if __name__ == '__main__':
    path = './data/input/'
    TRAIN_FILE = f'{path}train.csv'
    TEST_FILE = f'{path}test.csv'
    train = pd.read_csv(TRAIN_FILE, keep_default_na=False)
    test = pd.read_csv(TEST_FILE, keep_default_na=False)

    preprocess(train)
    preprocess(test)

    train.to_csv(f"{path}train_preprocessed.csv", index=False)
    test.to_csv(f"{path}test_preprocessed.csv", index=False)

