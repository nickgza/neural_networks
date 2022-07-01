from layers.layer import Layer
import numpy as np
from scipy import signal

class FCLayer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward_propagate(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.weights @ self.input + self.bias

    def backward_propagate(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class ConvLayer(Layer):
    def __init__(self, input_shape: tuple[int], kernel_size: int, depth: int) -> None:
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    
    def forward_propagate(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='valid')
        return self.output

    def backward_propagate(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], mode='full')

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape: tuple[int], output_shape: tuple[int]):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward_propagate(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward_propagate(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
