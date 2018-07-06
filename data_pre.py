"""
该代码是用keras判断短文本语义相似度
是用cnn和相似距离结合构建深度学习模型框架
"""

import os
import numpy as np
import pandas as pd
import pickle

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

def word_map(sentence,w):
    line =[]
    for word in sentence:
        if word in list(w.keys()):
            q = w[word]
        line.append(q)
    return line



"""
train map questions

"""
def train_map_question(df_train,questions):
    
    """
    df_train:training data
    questions:word_dict
    q1:sentence 1
    q2:sentence 2
    """

    train_dict_q1=[]
    for i in range(df_train.shape[0]):
        train_dict_q1.append(questions[df_train['q1'][i]])

    train_dict_q2=[]
    for j in range(df_train.shape[0]):
        train_dict_q2.append(questions[df_train['q2'][j]])

    return train_dict_q1,train_dict_q2


"""
test map questions

"""

def test_map_question(df_test,questions):

    """
    df_train:training data
    questions:word_dict
    q1:sentence 1
    q2:sentence 2
    """
    test_dict_q1=[]
    for i in range(df_test.shape[0]):
        test_dict_q1.append(questions[df_test['q1'][i]])
    test_dict_q2 =[]
    for j in range(df_test.shape[0]):
        test_dict_q2.append(questions[df_test['q2'][j]])
    return test_dict_q1,test_dict_q2


"""
语句拼接
"""

def sent1_merge_sent2(x1,x2):
    """
    x1:sentence 1 been split
    x2: sentence 2 been split
    """
    merge=[]
    for i in range(len(x1)):
        temp = x1[i]+[0]*(39-len(x1[i]))+x2[i]+[0]*(39-len(x2[i]))
        merge.append(temp)
    return merge

"""
ganerate train index list
"""

def generate_word_index(data,w):
    """
    data:拼接后的sentence index list
    """

    sent =[]
    flag = 0
    for s in data:
        flag = flag+1
        if flag % 10000 ==0:
            print("now on genearating %s"%flag)
        temp = word_map(s,w)
        sent.append(temp)

"""
generate word embedding
"""
def sentence_embedding(data,g):
    """
    data: words index been merged
    """
    emb_ag = []
    for se in data:
        ll = []
        for m in se:
            ll.append(g[m])
        emb_ag.append(ll)
    return emb_ag


"""
相似距离计算
"""
def eu_man_dis(val):
    eu_dis=[]
    man_dis=[]
    for i in range(len(val)):
        vector1 = mat(val[i][0:39].mean(axis=0))
        vector2 = mat(val[i][39:].mean(axis=0))
        eu_distance = np.sqrt(np.sum(np.square(vector1 - vector2))) #eu_dist
        man_distance = np.sum(np.abs(vector1-vector2)) #manha dis
        eu_dis.append(eu_distance)
        man_dis.append(man_distance)
    eu_dis = np.array(eu_dis)
    man_dis = np.array(man_dis)
    eu_man_dist = mat([eu_dis,man_dis]).T
    return eu_man_dist



"""
本数据集采用拍拍贷第三届魔镜杯脱敏的数据
备注：keras虽然方便好用，但是在处理自定义的神经网络并不是很灵活，故接下来将学习使用pytorch构建深度学习模型
"""





