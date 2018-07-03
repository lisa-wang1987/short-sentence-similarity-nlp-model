"""
该代码是用keras判断短文本语义相似度
是用cnn和相似距离结合构建深度学习模型框架
"""

import os
import numpy as np
import pandas as pd
from numpy import *

"""
keras model api
"""
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Input
from keras.layers import Embedding,multiply,subtract
from keras.layers import Conv1D,GlobalMaxPool1D,Lambda
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import Flatten
from keras import backend as K
from keras.models import Model
from keras.layers.core import Reshape
from keras.layers import Merge,concatenate
from keras.optimizers import RMSprop

"""
预处理函数
"""
def question(path):
	question = {}
	with open(path) as f:
		for line in f:
			l = line.strip('\n').split(',')
			key = l[0]
			value = l[1].split(' ')
			if key =='qid':
				continue
			else:
				question[key] = value
	que = list(question.keys())
	return que,question


def word_embedding(path):
    w = dict()
    w[0]=0
    g = []
    g.append(np.zeros(300))
    with open(path,'r') as f:
        for line in f:
            l = line.split()
            word = l[0]
            w[word] = len(g)
            g.append(np.array(l[1:]).astype(float))
    return w,g

def word_map(sentence):
    line =[]
    for word in sentence:
        if word in list(w.keys()):
            q = w[word]
        line.append(q)
    return line

"""
导入所需数据
"""

print('load questions...')
que,questions = question('question.csv')

print('load word and embeding list...')
w,g = word_embedding('word_embed.txt')

print('loading data...')
df_train= pd.read_csv('df_train.csv')
df_test = pd.read_csv('df_test.csv')
df_train = df_train
df_test = df_test

print('loading label....')
train_labels=np.array(df_train['label'])
test_labels = np.array(df_test['label'])



"""
特征整合
- 1.把两个sentence组合一起 embedding，对组合后的embediing cnn get maxpool
- 2.计算两个sentence 之间的相似距离
- 3.组合maxpool和相似距离
"""
"""
1.特征组合
"""

"""
sentence1 和 sentence2整合到一起 on training
"""

ag_train_s0x = list(df_train['words_1'])
ag_train_s1x = list(df_train['words_2'])
ag_train_s = []
for i in range(len(ag_train_s0x)):
    ag_train_a = ag_train_s0x[i].split(' ')+[0]*(39-len(ag_train_s0x[i].split(' ')))+ag_train_s1x[i].split(' ')+[0]*(39-len(ag_train_s1x[i].split(' ')))
    ag_train_s.append(ag_train_a)

"""
sentence1 和 sentence2整合到一起 on testing
"""
ag_test_s0x = list(df_test['words_1'])
ag_test_s1x = list(df_test['words_2'])
ag_test_s = []
for i in range(len(ag_test_s0x)):
    ag_test_a = ag_test_s0x[i].split(' ')+[0]*(39-len(ag_test_s0x[i].split(' ')))+ag_test_s1x[i].split(' ')+[0]*(39-len(ag_test_s1x[i].split(' ')))
    ag_test_s.append(ag_test_a)


print('generating ag train  index list')

ag_train_sent =[]
ag_train=0
for ag_sent in ag_train_s:
    ag_train = ag_train+1
    if ag_train%500==0:
        print(ag_train)
    ag_line2 = word_map(ag_sent)
    ag_train_sent.append(ag_line2)
    
print('generating ag test  index list')

ag_test_sent =[]
ag_test=0
for ag_sent in ag_test_s:
    ag_test = ag_test+1
    if ag_test%500==0:
        print(ag_test)
    ag_line3 = word_map(ag_sent)
    ag_test_sent.append(ag_line3)
    
    
"""
生成文本向量 ag
"""

print('training sentence embedding')
emb_train_ag=[]
for ss1 in ag_train_sent:
    l1 =[]
    for s1 in ss1:
        l1.append(g[s1])
    emb_train_ag.append(l1)

print('testing sentence embedding')
emb_test_ag=[]
for ss1 in ag_test_sent:
    l1 =[]
    for s1 in ss1:
        l1.append(g[s1])
    emb_test_ag.append(l1)

emb_train_ag = np.array(emb_train_ag)
emb_test_ag = np.array(emb_test_ag)

emb_train_ag = emb_train_ag.astype('float32')
emb_test_ag = emb_test_ag.astype('float32')

print('training data ag shape is :',emb_train_ag.shape)
print('test data ag shape is :',emb_test_ag.shape)




"""
相似距离计算
"""


def eu_man_dis(val):
    eu_dis=[]
    man_dis=[]
    for i in range(val.shape[0]):
        vector1 = mat(val[i][0:39].mean(axis=0))
        vector2 = mat(val[i][39:].mean(axis=0))
        eu_distance = np.sqrt(np.sum(np.square(vector1 - vector2))) #eu_dist
        man_distance = np.sum(np.abs(vector1-vector2)) #manha dis
        eu_dis.append(eu_distance)
        man_dis.append(man_distance)
    eu_dis = np.array(eu_dis)
    man_dis = np.array(man_dis)
    eu_man_dist = np.array(mat([eu_dis,man_dis]).T)
    return eu_man_dist
train_dis = eu_man_dis(emb_train_ag)
test_dis = eu_man_dis(emb_test_ag)



"""
keras model
"""

"""
超参数
"""
filters = 150
embedding_dims =300
hidden_dims =300
epochs =10
kernel_size=1
num_classes=2

emb_train_ag = emb_train_ag.astype('float32')
emb_test_ag = emb_test_ag.astype('float32')
y_train = keras.utils.to_categorical(train_labels,num_classes=2)
y_test = keras.utils.to_categorical(test_labels,num_classes=2)
x_train = emb_train_ag.reshape(emb_train_ag.shape[0],emb_train_ag.shape[1],emb_train_ag.shape[2],1)
x_test = emb_test_ag.reshape(emb_test_ag.shape[0],emb_test_ag.shape[1],emb_test_ag.shape[2],1)

train_dis = train_dis.astype('float32')
test_dis = test_dis.astype('float32')
x_train_dist = train_dis.reshape(train_dis.shape[0],1,1,train_dis.shape[1])
x_test_dist = test_dis.reshape(test_dis.shape[0],1,1,test_dis.shape[1])



visible1 = Input(shape=x_train.shape[1:])
conv1 = Conv2D(64,kernel_size=(1,300),activation='relu')(visible1)
pool1 = MaxPooling2D(pool_size=(78,1))(conv1)

conv2 = Conv2D(64,kernel_size=(2,300),activation='relu')(visible1)
pool2 = MaxPooling2D(pool_size=(77,1))(conv2)

"""
两个输出整合时会遇到整合报错的难题，注意lamda的应用
"""

visible2 = Input(shape=x_train_dist.shape[1:])

conv21 = Conv2D(64,kernel_size=(1,1),activation='relu')(visible2)
pool21 = MaxPooling2D(pool_size=(1,1))(conv21)

my_concat = Lambda(lambda x: K.concatenate([x[0],x[1],x[2]],axis=-1))
merge = my_concat([flat1,flat2,flat3])

#merge1 =Flatten()(merge)
hidden1 = Dense(10,activation='relu')(merge1)
output = Dense(2,activation='sigmoid')(hidden1)
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(x_train,y_train,epochs=30,verbose=1,validation_data=(x_test,y_test))
score = model.evaluate(x_test,y_test,verbose=0)
print('test loss:',score[0])
print('test accuracy:',score[1])



"""
本数据集采用拍拍贷第三届魔镜杯脱敏的数据
备注：keras虽然方便好用，但是在处理自定义的神经网络并不是很灵活，故接下来将学习使用pytorch构建深度学习模型
"""





