import sys
sys.path.append('..')
import numpy as np
from layers.layers import *
from layers.activation_funcs import *
from loss_funcs import *
from network import Network

# from keras.datasets import mnist
from keras.utils import np_utils
import pygame as pg

def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int | None = None):
    x = x.reshape(len(x), 28 * 28, 1).astype('float32') / 0xff
    y = np_utils.to_categorical(y).reshape(len(y), 10, 1)

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    return x, y

with np.load('mnist.npz') as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 30)

network = Network([
    FCLayer(28 * 28, 10),
    ReLU(),
    FCLayer(10, 10),
    Softmax(),
])

# Train
# network.train(mean_squared, d_mean_squared, x_train, y_train, epochs=500, learning_rate=0.05, every=50)

# # Test
# for ind, (x, y) in enumerate(zip(x_test, y_test)):
#     output = network.predict(x)
#     print(f'{ind}  pred: {np.argmax(output)},  true: {np.argmax(y)},  {"T" if np.argmax(output) == np.argmax(y) else "F"}  confidence: {round(output[np.argmax(output)][0] * 100, 3)}%')




# Display
def calculate_top_left(width, height):
    if width > height:
        return ((width - height) // 2, 0)
    else:
        return (0, (height - width) // 2)

def draw_squares(WIN, data: np.ndarray):
    '''Scales up by 255'''
    data = np.reshape(data, (28, 28)) * 255

    size = pg.display.get_surface().get_size()
    dim = min(size)
    square_dim = round(dim / 28)
    top_left_x, top_left_y = calculate_top_left(*size)

    for (y, x), val in np.ndenumerate(data):
        pg.draw.rect(WIN, (int(val),) * 3, pg.Rect(top_left_x + x * square_dim,
                                       top_left_y + y * square_dim,
                                       square_dim, square_dim))

def get_square(x: int, y: int) -> tuple[int, int]:
    size = pg.display.get_surface().get_size()
    dim = min(size)
    square_dim = round(dim / 28)
    top_left_x, top_left_y = calculate_top_left(*size)

    return ((y - top_left_y) / square_dim, (x - top_left_x) / square_dim)

def show_digit(data: int | np.ndarray = 0, mode='index'):
    ''' mode = 'index' | 'data' '''
    if isinstance(data, np.ndarray):
        mode = 'data'
    WIDTH, HEIGHT = 280, 280
    WIN = pg.display.set_mode((WIDTH, HEIGHT), pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
    pg.display.set_caption('')
    FPS = 30

    if mode == 'index':
        print(np.argmax(y_train[data]))
    RUN = True
    clock = pg.time.Clock()

    while RUN:
        clock.tick(FPS)
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    RUN = False

        WIN.fill(0)
        if mode == 'index':
            draw_squares(WIN, x_train[data])
        elif mode == 'data':
            draw_squares(WIN, data)

        pg.display.flip()

    pg.quit()

def draw_digit(draw_radius: int = 1) -> np.ndarray:
    WIDTH, HEIGHT = 280, 280
    WIN = pg.display.set_mode((WIDTH, HEIGHT), pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
    pg.display.set_caption('')

    data = np.zeros((28, 28))

    RUN = True
    clock = pg.time.Clock()
    mouse_down = False

    while RUN:
        clock.tick()
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    RUN = False
                case pg.MOUSEBUTTONDOWN:
                    drawn = set()
                    mouse_down = True
                case pg.MOUSEBUTTONUP:
                    mouse_down = False
                case pg.KEYDOWN:
                    if event.key == pg.K_r:
                        data = np.zeros((28, 28))

        # calculate changes
        if mouse_down:
            mouse_pos = get_square(*pg.mouse.get_pos())
            if not (0 <= mouse_pos[0] <= 27 and 0 <= mouse_pos[1] <= 27):
                continue
            
            for y in range(int(mouse_pos[0]) - draw_radius, int(mouse_pos[0]) + draw_radius + 2):
                if not 0 <= y <= 27:
                    continue
                for x in range(int(mouse_pos[1]) - draw_radius, int(mouse_pos[1]) + draw_radius + 2):
                    if not 0 <= x <= 27:
                        continue
                    dist = np.linalg.norm(((mouse_pos[0] - y), (mouse_pos[1] - x))) / np.linalg.norm((draw_radius, draw_radius))
                    data[(y, x)] = max(data[(y, x)], 1 - dist ** 1.5)

        WIN.fill(0)
        draw_squares(WIN, data)

        pg.display.flip()

    pg.quit()
    return data.reshape(784, 1)

def predict():
    data = draw_digit()
    output = network.predict(data)
    print(f'predicted: {np.argmax(output)}  confidence: {round(output[np.argmax(output)][0] * 100, 3)}%')

show = show_digit
draw = draw_digit
