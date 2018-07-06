
"""
遇到的问题：数据量庞大，没法统一转化成array，也没办法pickle.dump，故保留list，转化工作在训练过程中完成
"""

"""
1.得到所有的可以进入模型的数据
2.批次拆分
3.训练
4.预测
"""

import data_pre as dp
import numpy as np 
from numpy import *
import pandas as pd

"""
导入所需数据
"""

print('load questions...')
que,questions = dp.question('question.csv')

print('load word and embeding list...')
w,g = dp.word_embedding('word_embed.txt')

print('loading data...')
df_train= pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = df_train.head(10)
df_test = df_test.head(1)

print('loading label....')
train_labels=np.array(df_train['label'])


print('question map sentence,generate sentence list...')

#train_dict_q1,train_dict_q2 = dp.train_map_question(df_train,questions)
#test_dict_q1,test_dict_q2 = dp.test_map_question(df_test,questions)

print('sentence merge...')

#train_merge = dp.sent1_merge_sent2(train_dict_q1,train_dict_q2)
#test_merge = dp.sent1_merge_sent2(test_dict_q1,test_dict_q2)

print('generate word index after being merge...')

#train_words_index = dp.generate_word_index(train_merge,w)
#test_words_index = dp.generate_word_index(test_merge,w)


print('generate sentence embeding...')

#train_words_embedding = dp.sentence_embedding(train_words_index,g)
#train_words_embedding = np.array(train_words_embedding).astype('float32')
#test_words_embedding = dp.sentence_embedding(test_words_index,g)
t#est_words_embedding = np.array(test_words_embedding).astype('float32')

print('computing sentence similarity distance....')

#train_dist = dp.eu_man_dis(train_words_embedding)
#test_dist = dp.eu_man_dis(test_words_embedding)

with open('emb_train_ag.pkl','rb') as f:
	emb_train_ag = pickle.load(f)
with open('emb_test_ag.pkl','rb') as f:
	emb_test_ag = pickle.load(f)



with open('train_dist.pkl','rb') as f:
	train_dist = pickle.load(f)

with open('valid_dist.pkl','rb') as f:
	valid_dist = pickle.load(f)

print('now i have got train word embedding and sentence similarity distance,the following work will be build netword and go on training,fighting!')


"""
生成复合入模格式的数据
"""

print('training list to array...')
"""
这么麻烦的原因是：pickle只能存储低于4GB，要切片存储
"""

train_array = np.array(emb_train_ag[0:10000]).astype('float32')
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[10000:20000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[20000:30000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[30000:40000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[40000:50000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[50000:60000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[60000:70000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[70000:80000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[80000:90000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[90000:100000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[100000:110000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[110000:120000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[120000:130000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[130000:140000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[140000:150000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[150000:160000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[160000:170000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[170000:180000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[180000:190000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[190000:200000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[200000:210000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[210000:220000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[220000:230000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[230000:240000]).astype('float32'),axis=0)
train_array = np.insert(train_array,len(train_array),values=np.array(emb_train_ag[240000:250000]).astype('float32'),axis=0)

print('valid list to array...')

valid_array = np.array(emb_train_ag[250000:]).astype('float32')

print('testing list to array ...')

test_array = np.array(emb_test_ag[0:20000]).astype('float32')
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[20000:40000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[40000:60000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[60000:80000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[80000:100000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[100000:120000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[120000140000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[140000:160000]).astype('float32'),axis=0)
test_array = np.array(test_array,len(test_array),values=np.array(emb_test_ag[160000:]).astype('float32'),axis=0)


print('similarity distance to array...')

train_dist = np.array(train_dist).astype('float32')
valid_dist = np.array(valid_dist).astype('float32')

"""
Keras Model buliding
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
超参数
"""
filters = 150
embedding_dims =300
hidden_dims =30
epochs =20
kernel_size=1
num_classes=2


"""
reshape
"""

y_train = keras.utils.to_categorical(train_labels[0:200000],num_classes=2)
y_valid = keras.utils.to_categorical(train_labels[200000:],num_classes=2)

x_train = train_array.reshape(train_array.shape[0],train_array.shape[1],train_array.shape[2],1)
x_valid = valid_array.reshape(valid_array.shape[0],valid_array.shape[1],valid_array.shape[2],1)


train_dist = train_dist.reshape(train_dist.shape[0],1,train_dist.shape[1],1)
valid_dist = valid_dist.reshape(valid_dist.shape[0],1,valid_dist.shape[1],1)


"""
model
"""

visible_1 = Input(shape = x_train.shape[1:])
conv_1 = Conv2D(64,kernel_size=(1,300),activatio='relu')(visible_1)
pool_1_max = MaxPooling2D(pool_size=(78,1))(conv_1)
pool_1_avg = AveragePooling2D(pool_size=(1,1))(conv_1)

flat1_max = Flatten()(pool_1_max)
flat1_avg = Flatten()(pool_1_avg)

conv_2 = Conv2D(64,kernel_size=(2,300),activation = 'relu')(visible_1)
pool_2_max = MaxPooling2D(pool_size=(77,1))(conv_2)
pool_2_avg = AveragePooling2D(pool_size=(1,1))(conv_2)

flat2_max = Flatten()(pool_2_max)
flat2_avg = Flatten()(pool_2_avg)

visible_2 = Input(shape = train_dist.shape[1:])
conv_3 = Conv2D(64,kernel_size=(1,1),activation='relu')(visible_2)
pool_3_max = MaxPooling2D(pool_size=(1,1))(conv_3)
pool_3_avg = AveragePooling2D(pool_size=(1,1))(conv_3)

flat3_max = Flatten()(pool_3_max)
flat3_avg = Flatten()(pool_3_avg)

my_concat = Lambda(lambda x: K.concatenate([x[0],x[1],x[2],x[3],x[4],x[5]],axis=-1))

merge = my_concat([flat1_max,flat1_avg,flat2_max,flat2_avg,flat3_max,flat3_avg])

hidden = Dense(10,activation='relu')(merge)
output = Dense(2,activation='softmax')(hidden)

model = Model(inputs = [visible_1,visible_2],outputs=output)

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit([x_train,train_dist],y_train,epochs=epochs,verbose=1,validation_data=[x_valid,valid_dist],y_valid)
score = model.evaluate([x_valid,valid_dist],y_valid,verbose=0)
print('test score:',score[0])
print('test accuracy:',score[1])















