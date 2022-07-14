import sys
sys.path.append('..')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from tensorflow import keras
import numpy as np
import pygame as pg
from keras.utils import np_utils

def preprocess_data(x: np.ndarray, y: np.ndarray,):
    x = x.reshape(*x.shape, 1).astype('float32') / 0xff
    y = np_utils.to_categorical(y)

    return x, y

with np.load('mnist.npz') as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

early_stopping = keras.callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])

# Train
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=3,
    callbacks=[early_stopping],
)

# Evaluate
model.evaluate(x_test,  y_test, verbose=2)




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
    return data.reshape(1, 28, 28, 1)

def predict():
    data = draw_digit()
    output = model.predict(data)
    print(f'predicted: {np.argmax(output)}  confidence: {round(output[0][np.argmax(output)] * 100, 3)}%')

show = show_digit
draw = draw_digit
