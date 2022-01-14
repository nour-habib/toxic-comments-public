import csv
import pandas as pd
import torch
import numpy as np

import matplotlib.pyplot as pl
from pandas import DataFrame

train_data = pd.read_csv("train.csv",encoding="UTF-8")
test_data = pd.read_csv("test.csv",encoding="UTF-8")

print("Number of records in train_data: ")
print(len(train_data))

print("Number of records in test_data: ")
print(len(test_data))

#train_tensor = torch.Tensor(train_data.to_numpy())
# test_tensor = torch.Tensor(test_data.values)


#print(train_data)
#print(test_data)

tox_num = train_data['toxic'].value_counts().tolist()
sev_num = train_data['severe_toxic'].value_counts().tolist()
obs_num = train_data['obscene'].value_counts()
thr_num = train_data['threat'].value_counts()
ins_num = train_data['insult'].value_counts()
id_num = train_data['identity_hate'].value_counts()


#print(tox_num[1])
# print(sev_num[1])
# print(obs_num[1])
# print(thr_num[1])
# print(ins_num[1])
# print(id_num[1])

dt = {'Counts': [tox_num[1],sev_num[1],obs_num[1],thr_num[1],ins_num[1],id_num[1]]}
#print(dt)

fr = DataFrame(dt,columns=['Counts'],index=['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate'])



fr.plot.pie(y='Counts',figsize=(5,5),autopct='%1.1f%%',startangle=90)

pl.legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), ncol=2)
#pl.show()
#pl.savefig('countspie.png')


#check for unlabeled data

#get length of comments, get average, plot, etc

comment_df = pd.DataFrame(train_data['comment_text'])
comment_df_t =  pd.DataFrame(test_data['comment_text'])

comment_len = pd.DataFrame([[]])
comment_len_t = pd.DataFrame([[]])

for row in comment_df:
	#print(comment_df[row].apply(len))
	comment_len = comment_df[row].apply(len)

for row in comment_df_t:
	#print(comment_df[row].apply(len))
	comment_len_t = comment_df_t[row].apply(len)	

avg_len = comment_len.mean()
print("Average input length for training data: ") #prior to cleaning the data
print(avg_len)

avg_len_t = comment_len_t.mean()
print("Average input length for testing data: ")
print(avg_len_t)


unlabeled = train_data[(train_data['toxic']!=1)& (train_data['severe_toxic']!=1) & (train_data['obscene']!=1) & 
			(train_data['threat']!=1) & (train_data['insult']!=1) & (train_data['identity_hate']!=1)]
#print(len(unlabeled)/len(train_data)*100)



