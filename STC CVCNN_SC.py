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


import sklearn
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import manifold
from scipy.optimize import linear_sum_assignment as linear_assignment


def visualizeData(Z, labels, num_clusters, title):

    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=1, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)


# 聚类精度计算
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def get_cluster_metric_string(labels_true, labels_pred):
    '''
    Creates a formatted string containing the method name and acc, nmi metrics - can be used for printing
    :param labels_true: True label for each sample
    :param labels_pred: Predicted label for each sample
    '''
    acc = cluster_acc(labels_true, labels_pred)
    nmi = sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred)

    ars = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
    vms = sklearn.metrics.v_measure_score(labels_true, labels_pred)
    print(nmi)
    print(acc)
    print(ars)
    print(vms)


def get_silhouette(X, labels_pred):
    ss = sklearn.metrics.silhouette_score(X, labels_pred)
    print(ss)

n_classes = 10
X_test, Y_test = TestDataset(n_classes)
X_test = (X_test - min_value)/(max_value - min_value)
X_test_feature = base_model.predict(X_test,verbose=0)

from sklearn.cluster import KMeans
tsne = manifold.TSNE(n_components=2)  # init='pca'
Z_tsne = tsne.fit_transform(X_test_feature)
Y_pred = KMeans(n_clusters=n_classes).fit_predict(Z_tsne)
get_cluster_metric_string(Y_test, Y_pred)
get_silhouette(X_test_feature, Y_test)