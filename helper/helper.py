import pickle
from sklearn.utils import shuffle
import numpy as np


def load_data():
    with open('traffic-signs-data/train.p', mode='rb') as train_data:
        train = pickle.load(train_data)
    with open('traffic-signs-data/test.p', mode='rb') as test_data:
        test = pickle.load(test_data)
    with open('traffic-signs-data/valid.p', 'rb') as valid_data:
        valid = pickle.load(valid_data)
    return train, valid, test


def split_data(data):
    x_train, y_train = data['features'], data['labels']
    return x_train, y_train


def shuffle_data(x, y):
    X, Y = shuffle(x, y)
    return X, Y


def to_grayscale(x, axis):
    x = np.sum(x, axis=axis, keepdims=True)
    return x


def normalize(x, mean, std):
    x = (x - mean) / std
    return x
