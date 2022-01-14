import optparse
import csv
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
pd.options.mode.chained_assignment = None  
def modelMetrics(inFile,outFile,isClass):
    labels = pd.read_csv(inFile,encoding="UTF-8")
    test_labels = labels[labels['threat'].map(lambda x: x>-1)]
    #print(len(test_labels))
    plabels = pd.read_csv(outFile,encoding="UTF-8")
    plabels['threat'].to_csv("threats")
    pred_labels=plabels[plabels['id'].isin(test_labels['id'])]
    #print(len(pred_labels))
    class_names = list(labels.columns)[1:] #['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    #print(class_names)
    if(isClass==False):
        pred_labels[class_names]=np.where(pred_labels[class_names] > 0.5, 1, 0)

    for class_name in class_names:
        accuracy = 0
        prediction = pred_labels[class_name].tolist()
        actual = test_labels[class_name].tolist()
        for idx in range(len(test_labels)):
            if (actual[idx] == 1 and prediction[idx] == 1) or (actual[idx] == 0 and prediction[idx] == 0):
                accuracy = accuracy + 1
        
        print(f"Accuracy for {class_name}: {accuracy/len(test_labels)}")

    report = classification_report(test_labels[class_names],pred_labels[class_names],target_names=class_names,zero_division=False)
    print(report)


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputFile", dest="trueLabelFile", default="test_labels.csv", help="True labels of dataset")
    optparser.add_option("-o", "--outputFile", dest="outputLabelFile", default="LSTM.csv", help="Predicted scores for dataset")
    optparser.add_option("-c", "--classes", dest="isClass",default = False, help="True if output csv contains binary class information instead of confidence scores(default false)")
    optparser.add_option("-a", "--all", dest="allModels", default=False, action = "store_true", help="Print metrics for All ML models")

    (opts, _) = optparser.parse_args()
    trueLabelFile = "data/input/"+opts.trueLabelFile
    isClass = opts.isClass

    if(opts.allModels):
        models = [m for m in listdir("data/output") if isfile(join("data/output/", m))]
        for model in models:
            print(f"Metrics for {model}\n")
            predLabelFile = join("data/output/",model) 
            modelMetrics(trueLabelFile, predLabelFile, isClass)
            print("\n")
    else:    
        predLabelFile = "data/output/"+opts.outputLabelFile    
        modelMetrics(trueLabelFile, predLabelFile, isClass)
