import numpy as np


def get_activation_func(key):
    activation_func = {
        'relu': [relu, relu_derivative],
        'sigmoid': [sigmoid, sigmoid_derivative],
        'linear': [linear, linear_derivative],
        'tanh': [tanh, tanh_derivative],
        'softmax': [softmax, softmax_derivative]
    }
    return activation_func[key] if key is not None else [None, None]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def linear(x):
    return x


def linear_derivative(_):
    return 1


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_derivative(x):
    return x * (1 - x)

