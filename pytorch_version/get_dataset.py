import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch

def load_pretrain_dataset(num):
    x = np.load(f"/data/Dataset_WiFi_maml/X_train_{num}Class_run1.npy")
    y = np.load(f"/data/Dataset_WiFi_maml/Y_train_{num}Class_run1.npy")
    y = y.astype(np.uint8)
    return x, y

def load_train_dataset_k_shot(num, k_shot):
    x = np.load(f"/data/Dataset_WiFi_maml/X_train_{num}Class_run1.npy")
    y = np.load(f"/data/Dataset_WiFi_maml/Y_train_{num}Class_run1.npy")
    y = y.astype(np.uint8)
    random_index_shot = []
    for i in range(num):
        index_shot = [index for index, value in enumerate(y) if value == i]
        random_index_shot += random.sample(index_shot, k_shot)
    random.shuffle(random_index_shot)
    x_train_k_shot = x[random_index_shot, :, :]
    y_train_k_shot = y[random_index_shot]
    return x_train_k_shot, y_train_k_shot

def load_test_dataset(num):
    x1 = np.load(f"/data/Dataset_WiFi_maml/X_test_{num}Class_run1.npy")
    y1 = np.load(f"/data/Dataset_WiFi_maml/Y_test_{num}Class_run1.npy")
    y1 = y1.astype(np.uint8)

    x2 = np.load(f"/data/Dataset_WiFi_maml/X_test_{num}Class_run2.npy")
    y2 = np.load(f"/data/Dataset_WiFi_maml/Y_test_{num}Class_run2.npy")
    y2 = y2.astype(np.uint8)

    return x1, y1, x2, y2