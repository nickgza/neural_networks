import sys
sys.path.append('..')
import numpy as np
from layers.layers import *
from layers.activation_funcs import *
from loss_funcs import *
from network import Network

from keras.datasets import mnist
from keras.utils import np_utils
import pygame as pg

def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int | None = None):
    x = x.reshape(len(x), 28 * 28, 1).astype('float32') / 0xff
    y = np_utils.to_categorical(y).reshape(len(y), 10, 1)

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 30)

# network = Network([
#     FCLayer(28 * 28, 10),
#     ReLU(),
#     FCLayer(10, 10),
#     Softmax(),
# ])

# # train
# network.train(mean_squared, d_mean_squared, x_train, y_train, epochs=500, learning_rate=0.05, every=50)

# # test
# for x, y in zip(x_test, y_test):
#     output = network.predict(x)
#     print(f'pred: {np.argmax(output)},  true: {np.argmax(y)},  {"T" if np.argmax(output) == np.argmax(y) else "F"}')

# display

WIDTH, HEIGHT = 280, 280
WIN = pg.display.set_mode((WIDTH, HEIGHT), pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
pg.display.set_caption('Image')
FPS = 30

def calculate_top_left(width, height):
    if width > height:
        return ((width - height) // 2, 0)
    else:
        return (0, (height - width) // 2)

def draw_squares(WIN, data: np.ndarray):
    data = np.reshape(data, (28, 28)) * 255

    size = pg.display.get_surface().get_size()
    dim = min(size)
    square_dim = round(dim / 28)
    top_left_x, top_left_y = calculate_top_left(*size)

    for (y, x), val in np.ndenumerate(data):
        pg.draw.rect(WIN, (int(val), int(val), int(val)), pg.Rect(top_left_x + x * square_dim,
                                       top_left_y + y * square_dim,
                                       square_dim, square_dim))

RUN = True
clock = pg.time.Clock()

while RUN:
    clock.tick(FPS)
    for event in pg.event.get():
        match event.type:
            case pg.QUIT:
                RUN = False

    WIN.fill(0)
    draw_squares(WIN, x_train[10])
    print(np.argmax(y_train[10]))
    # if borders:
    #     draw_borders()

    pg.display.flip()

pg.quit()
