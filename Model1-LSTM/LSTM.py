import pickle
import pandas as pd
from preprocessing.dataset import Tokenizer, preprocess
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import torchtext
from collections import Counter
import numpy as np
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

train = Tokenizer("Yelp_train.csv")
test = Tokenizer("Yelp_test.csv")

lst = array(train.data['labelA'])
encoded = to_categorical(lst)
print(encoded)
# inverted = argmax(encoded[0])
# print(inverted)

# train.data['h_e'] = encoded

# lst=[]
# for i in range(200):
#   if test.data['labelA'][i]==3:
#     lst.append([0,0,1])
#   elif test.data['labelA'][i]==2:
#     lst.append([0,1,0])
#   elif test.data['labelA'][i]==1:
#     lst.append([1,0,0])

# test.data['h_e']=lst

lst = array(test.data['labelA'])
t_encoded = to_categorical(lst)
# test.data['h_e'] = encoded

train_mapping, inv_train_mapping = train.preprocess()
test_mapping, inv_test_mapping = test.preprocess()

# new_data = open('/content/drive/MyDrive/GCDC_rerelease/mapped_tokens_Yelp_train.csv.pkl','rb')
# new_t_data = open('/content/drive/MyDrive/GCDC_rerelease/mapped_tokens_Yelp_test.csv.pkl','rb')
# train_mapping = pickle.load(new_data)
# test_mapping = pickle.load(new_t_data)

len(train_mapping)
# train_mapping
train.data['encoding'] = train_mapping
test.data['encoding'] = test_mapping

np.random.seed(7)
X_train = sequence.pad_sequences(train.data['encoding'], maxlen=500)
y_train = encoded  # train.data['h_e']
X_test = sequence.pad_sequences(test.data['encoding'], maxlen=500)
y_test = t_encoded  # test.data['h_e']

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(30000, embedding_vector_length, input_length=500))
model.add(LSTM(4))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=40, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", (scores[1]*100))
