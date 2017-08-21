import numpy as np
from keras.datasets import cifar10


def cifar10_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def cifar10_data():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)