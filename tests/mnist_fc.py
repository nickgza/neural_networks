import sys
sys.path.append('..')
import numpy as np
from layers.layers import *
from layers.activation_funcs import *
from loss_funcs import *
from network import Network

from keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot as plt

def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int | None = None):
    x = x.reshape(len(x), 28 * 28, 1).astype('float32') / 0xff
    y = np_utils.to_categorical(y).reshape(len(y), 10, 1)

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 50)

network = Network([
    FCLayer(28 * 28, 10),
    ReLU(),
    FCLayer(10, 10),
    Softmax(),
])

# train
network.train(mean_squared, d_mean_squared, x_train, y_train, epochs=500, learning_rate=0.05, every=50)

# test
for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print(f'pred: {np.argmax(output)},  true: {np.argmax(y)},  {"T" if np.argmax(output) == np.argmax(y) else "F"}')
