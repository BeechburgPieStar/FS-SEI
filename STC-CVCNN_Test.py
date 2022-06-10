# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:44:04 2021

@author: 15851
"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config))  
import numpy as np
from keras.models import Model
from numpy import array
from keras.utils import to_categorical
import keras.models as models
from keras.layers import *
import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
from unit.complexnn.conv import ComplexConv1D
from unit.complexnn.bn import ComplexBatchNormalization
from unit.complexnn.dense import ComplexDense
from keras.layers import Input, Add, MaxPooling1D, Activation,Dense,Conv1D,BatchNormalization
from keras.models import Model
import keras
from keras.layers.core import Dropout,Flatten
from unit.triplet_losses import batch_all_triplet_loss
from unit.triplet_metrics import triplet_accuracy
import random

def TrainDataset(num):
    x = np.load(f"Dataset/X_train_{num}Class.npy")
    y = np.load(f"Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.1, random_state= 30)
    return X_train, X_val, Y_train, Y_val

def TrainDatasetKshotRround(num, k):
    x = np.load(f"Dataset/X_train_{num}Class.npy")
    y = np.load(f"Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)

    List_train = y.tolist()

    X_train_K_Shot = np.zeros([int(k*num), 6000, 2])
    Y_train_K_Shot = np.zeros([int(k*num)])

    for i in range(num):
        index_train_start = List_train.index(i)
        if i == num-1:
            index_train_end = y.shape[0]
        else:
            index_train_end = List_train.index(i+1)-1
        index_shot = range(index_train_start, index_train_end)
        random_shot = random.sample(index_shot, k)

        X_train_K_Shot[i*k:i*k+k,:,:] = x[random_shot,:,:]       
        Y_train_K_Shot[i*k:i*k+k] = y[random_shot]
    return X_train_K_Shot, Y_train_K_Shot

def TestDataset(num):
    x = np.load(f"Dataset/X_test_{num}Class.npy")
    y = np.load(f"Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y

def base_model():
    x_input = Input(shape=(6000, 2))
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x_input)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = ComplexConv1D(64, 3, activation='relu', padding='same')(x)
    x = ComplexBatchNormalization()(x)
    x = MaxPooling1D(pool_size= 2)(x)
    x = Flatten()(x)
    embedding = Dense(1024, activation='relu')(x)
    # embedding = BatchNormalization()(x)
    return Model(x_input, embedding)

base_model = base_model()

classifier_output = Dense(90)(base_model.outputs[-1])
classifier_output = Activation('softmax', name='Classifier')(classifier_output)

input_target = Input(shape=(1,))
centers = Embedding(90, 1024)(input_target)
center_output = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='Center')([base_model.outputs[-1],centers])

margin = 5
lambda_T = 0.01
lambda_C = 0.01
model = Model(inputs=[base_model.inputs[0],input_target],
    outputs=[classifier_output,base_model.outputs[-1],center_output])     

model.compile(loss=["categorical_crossentropy", batch_all_triplet_loss,lambda y_true,y_pred: y_pred], loss_weights = [1, lambda_T, lambda_C], optimizer='adam', metrics=["acc"])

X_train, X_val, value_Y_train, value_Y_val = TrainDataset(90)
min_value = X_train.min()
min_in_val =  X_val.min()
if min_in_val < min_value:
    min_value = min_in_val

max_value = X_train.max()
max_in_val =  X_val.max()
if max_in_val > max_value:
    max_value = max_in_val

X_train = (X_train - min_value)/(max_value - min_value)
X_val = (X_val- min_value)/(max_value - min_value)

Y_train = to_categorical(value_Y_train)
Y_val = to_categorical(value_Y_val)

model.load_weights(f"Model/STC CVCNN_lambda={lambda_T}_m={margin}.hdf5")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from warnings import filterwarnings
filterwarnings('ignore')

Ks = [1, 5, 10, 15, 20]#
num_Ks = 1
Ns = [10, 20, 30]
num_Ns = 3
Rs = 1000
acc = np.zeros([int(num_Ks*num_Ns), Rs])

import time
for r in range(Rs):
    t1 = time.time()
    print(f"--------r={r}---------")
    for n in range(num_Ns):
        X_test, Y_test = TestDataset(Ns[n])
        X_test = (X_test - min_value)/(max_value - min_value)
        X_test_feature = base_model.predict(X_test,verbose=0)
        for k in range(num_Ks):
            x, y = TrainDatasetKshotRround(Ns[n], Ks[k])
            x = (x - min_value)/(max_value - min_value)
            x_feature  = base_model.predict(x)
            clf = KNeighborsClassifier()#KNeighborsClassifier()
            clf.fit(x_feature, y)            
            acc[n*num_Ks + k, r] = clf.score(X_test_feature, Y_test)
    t2 = time.time()
    print(t2 - t1)

import pandas as pd
df = pd.DataFrame(acc)
df.to_excel(f"Result/STC CVCNN_lambda={lambda_T}_m={margin}.xlsx", index=False)

acc_3m = np.zeros([int(num_Ks*num_Ns), 3])
acc_3m[:, 0] = np.mean(acc, axis = 1)
acc_3m[:, 1] = np.max(acc, axis = 1)
acc_3m[:, 2] = np.min(acc, axis = 1)

df = pd.DataFrame(acc_3m, columns=['mean','max', 'min'])
df.to_excel(f"Result/STC CVCNN_lambda={lambda_T}_m={margin}_3m.xlsx", index=False)