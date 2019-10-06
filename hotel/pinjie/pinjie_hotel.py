
from gensim.models import  KeyedVectors
from  sklearn.metrics import  accuracy_score
from sklearn.svm import LinearSVC
from  sklearn.metrics import recall_score
from  sklearn.metrics import precision_score
from  sklearn.metrics import f1_score
from sklearn import  svm
import tensorflow as tf



f = open("..\data\labels.txt", 'r', encoding='utf-8')
rr = f.readlines()
d = {}
for r in rr:
	id = r.split('\t')[0]
	flag = r.split('\t')[-1].split('\n')[0]
	if flag == '1':
		d[id] = 1
	else:
		d[id] = 0

f1 = open("..\data\ColdStart_Update/train.txt", 'r', encoding='utf-8')
tt = f1.readlines()
train = []
for t in tt:
	train.append(t.split('\n')[0])

labels1 = []
for i, t in enumerate(train):
		# print(i)
	labels1.append(d[t])

f2 = open("..\data\ColdStart_Update/test.txt", 'r', encoding='utf-8')
ttt = f2.readlines()
test = []
for t in ttt:
	test.append(t.split('\n')[0])

labels2 = []
for t in test:
	labels2.append(d[t])


model1 = KeyedVectors.load_word2vec_format(
    "bfbyrd.txt",
    binary=False)
model2 = KeyedVectors.load_word2vec_format(
    "bfbytime.txt",
    binary=False)

model3 = KeyedVectors.load_word2vec_format(
    "bfbytext.txt",
    binary=False)


bf1 = [model1[d] for d in train]   #by text  train
bf2 = [model1[d] for d in test]    #test

bf3 = [model2[d] for d in train]   #by ratio  train
bf4 = [model2[d] for d in test]    #test

bf5 = [model3[d] for d in train]  #by time train
bf6 = [model3[d] for d in test]   #test



##########arrary 转 list


for i in range(len(train)):
	bf1[i] = bf1[i].tolist()

for i in range(len(train)):
	bf3[i] = bf3[i].tolist()

for i in range(len(train)):
	bf5[i] = bf5[i].tolist()

for i in range(len(test)):
	bf2[i] = bf2[i].tolist()
for i in range(len(test)):
	bf4[i] = bf4[i].tolist()
for i in range(len(test)):
	bf6[i] = bf6[i].tolist()

############
a1=[[0.39795583,0.1252339,0.0892195,0.71981955,0.5375913,.6784933 ]]
a3=[[30.014595,30.636536,30.735222,30.692133,30.310194,30.632332]]
a5= [[0.25417903,1.0097831,0.19205137,0.5019999,0.56852525,0.43275815]]

import numpy as np
z1=np.multiply(a1,bf1)
z2=np.multiply(a3,bf3)
z3=np.multiply(a5,bf5)


z4=np.multiply(a1,bf2)
z5=np.multiply(a3,bf4)
z6=np.multiply(a5,bf6)

z10=z1+z2+z3
z11=z4+z5+z6
print(len(z10[0]))


f3=open("train_labels.txt",'w',encoding='utf-8')
f4=open("test_labels.txt",'w',encoding='utf-8')

for i,ll in enumerate(labels1):
	f3.write(str(ll))
	f3.write('\n')

for i,ll in enumerate(labels2):
	f4.write(str(ll))
	f4.write('\n')



clf = svm.SVC(kernel='linear', C=1)
clf.fit(z10, labels1)
predicted = clf.predict(z11)

acc = accuracy_score(labels2, predicted)
recall = recall_score(labels2, predicted)
pre = precision_score(labels2, predicted)
f1 = f1_score(labels2, predicted)
print("pre=", pre)
print("recall=", recall)
print("f1=", f1)
print("acc=", acc)
print('\n')
