# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:44:04 2021

@author: 15851
"""
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

def TrainDataset(num):
    x = np.load(f"Dataset/X_train_{num}Class.npy")
    y = np.load(f"Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.1, random_state= 30)
    return X_train, X_val, Y_train, Y_val

def TestDataset(num):
    x = np.load(f"Dataset/X_test_{num}Class.npy")
    y = np.load(f"Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y

def base_model():
    x_input = Input(shape=(4800, 2))
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


checkpoint = ModelCheckpoint(f"Model/STC-CVCNN_lambda={lambda_T}_m={margin}.hdf5", #_{submodel}submodel = 15
                             verbose=1, 
                             save_best_only=True)
model.fit([X_train,value_Y_train], [Y_train, value_Y_train,value_Y_train],
          shuffle = True,
          validation_data=([X_val,value_Y_val], [Y_val, value_Y_val,value_Y_val]),
          batch_size=32,
          epochs=200,
          verbose=2,
          callbacks=[checkpoint])