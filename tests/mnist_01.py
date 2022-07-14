import sys
sys.path.append('..')
import numpy as np
from layers.layers import *
from layers.activation_funcs import *
from loss_funcs import *
from network import Network

# from keras.datasets import mnist
from keras.utils import np_utils

def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int | None = None):
    zero_index = np.where(y == 0)[0]
    one_index = np.where(y == 1)[0]
    if limit is not None:
        zero_index = zero_index[:limit]
        one_index = one_index[:limit]
    
    all_indices = np.concatenate((zero_index, one_index))
    all_indices = np.random.default_rng().permutation(all_indices)

    x, y = x[all_indices], y[all_indices]

    x = x.reshape(len(x), 1, 28, 28).astype('float32') / 0xff
    y = np_utils.to_categorical(y).reshape(len(y), 2, 1)

    return x, y

with np.load('mnist.npz') as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 30)

network = Network([
    ConvLayer((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    FCLayer(5 * 26 * 26, 100),
    Sigmoid(),
    FCLayer(100, 2),
    Softmax(),
])

# Train
network.train(binary_cross_entropy, d_binary_cross_entropy, x_train, y_train, epochs=20, learning_rate=0.05)

# Test
for ind, (x, y) in enumerate(zip(x_test, y_test)):
    output = network.predict(x)
    print(f'{ind}  pred: {np.argmax(output)},  true: {np.argmax(y)},  {"T" if np.argmax(output) == np.argmax(y) else "F"}  confidence: {round(output[np.argmax(output)][0] * 100, 3)}%')
