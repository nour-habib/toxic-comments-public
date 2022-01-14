import csv
import pandas as pd
import torch
import numpy as np
import re

import matplotlib.pyplot as pl
from pandas import DataFrame

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack



train_data = pd.read_csv("train.csv",encoding="UTF-8")
test_data = pd.read_csv("test.csv",encoding="UTF-8")
test_labels = pd.read_csv("test_labels.csv",encoding="UTF-8")

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

#print(train_data['comment_text'])

#clean() borrowed from https://www.kaggle.com/rhodiumbeng/classifying-multi-label-comments-0-9741-lb

def clean(text):
    text = text.lower()
    text = re.sub(r'[?|!|\'|"|#]',r'',text)
    text = re.sub(r'[.|,|)|(|\|/]',r' ',text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'youll", " you will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

comment_tr = train_data['comment_text'] = train_data['comment_text'].map(lambda com : clean(com))
#print(comment_tr)

comment_test = test_data['comment_text'] = test_data['comment_text'].map(lambda com : clean(com))
# print(train_data.shape)


y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
x =  comment_tr.tolist()


cc_nb = ClassifierChain(MultinomialNB(alpha=1.0)) #default alpha=1.0 w/ Laplace Smoothing
tfidVect = TfidfVectorizer(stop_words='english')



print("Results for Naive Bayes Model on validation set... ")
#Results on test data outputed to nb_test_pred.csv


#roc graph
pl.figure(0,figsize=(8,8)).clf()
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")

i = 1
testP = pd.DataFrame(test_data['id'])
#print(testP)

for l in labels:
    y = train_data[[l]]
    y_vals = y.values
    X_train, X_test, y_train, y_test = train_test_split(comment_tr, y_vals, test_size=0.33, random_state=53)
    #print(X_train.shape)
    #print(y_vals)


    countVect = CountVectorizer(stop_words='english')

    countVect_train = countVect.fit_transform(X_train)
    countVect_test = countVect.transform(X_test)

    tfidVect.fit_transform(X_train)
    tfidVect.transform(X_test)

    cc_nb.fit(countVect_train,y_train)

    pred = cc_nb.predict(countVect_test)
    #print(pred)

    cv = countVect.transform(comment_test)
    tfidVect.transform(comment_test)
    p = cc_nb.predict(cv)
    
    print(l)
   

    #print(len(p))
    testP.insert(i,l,p,True)
    i=i+1

    score = accuracy_score(y_test,pred)
   
    print("Accuracy score: "+str(score))

    f1_scores = f1_score(y_test,pred,average='weighted') #y_true,y_pred
    print("F1 score: "+str(f1_scores))

    prec_score = precision_score(y_test,pred,average='weighted')
    print("Precision: "+str(prec_score))

    recall = recall_score(y_test,pred,average='weighted')
    print("Recall: "+str(recall))

    roc_score = roc_auc_score(y_test,pred)
    print("Roc score: "+str(roc_score)+"\n")
    print('Log loss: for '+l+' is {}'.format(log_loss(y_test,pred)))
    false_p, true_p, threshold = roc_curve(y_test,pred)
    # con_matrix = confusion_matrix(y_test,pred)
    # print("Confusion matrix: "+str(con_matrix))
    # #print(classification_report(y_test, pred))
    # pl.plot(false_p,true_p,label=l+" AUC = "+str(roc_score))
    # pl.legend(loc="lower right")
    

#print(testP)
fc = testP.columns[0]
testP = testP.drop([fc],axis=1)
testP.to_csv("nb_test_pred.csv",index=False)

# pl.title("Naive Bayes")
# pl.show()
#pl.savefig('roc.png')



