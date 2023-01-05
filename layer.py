import numpy as np

from activation import *


class Layer:
    def __init__(self, units=None, activation=None, bias_initializer='zeros', weight_initializer='zeros'):

        self.shape = [0, units]
        self.activation, self.activation_derivative = get_activation_func(activation)
        self.weights = None
        self.bias = None
        self.input = None
        self.output = None
        self.bias_initializer = bias_initializer
        self.weight_initializer = weight_initializer
        self.delta = None

    def build(self, input_dim):
        self.shape[0] = input_dim
        if self.bias_initializer == 'random_uniform':
            self.bias = np.random.randn(self.shape[1]).reshape(1, -1)
        elif self.bias_initializer == 'ones':
            self.bias = np.ones(self.shape[1]).reshape(1, -1)
        else:
            self.bias = np.zeros(self.shape[1]).reshape(1, -1)

        if self.weight_initializer == 'glorot_uniform':
            limit = 6.0 / np.sqrt((self.shape[0] + self.shape[1]))
            self.weights = np.random.uniform(-limit, limit, size=self.shape)
        elif self.weight_initializer == 'random_uniform':
            self.weights = np.random.randn(self.shape[0], self.shape[1])
        elif self.weight_initializer == 'ones':
            self.weights = np.ones(shape=self.shape)
        else:
            self.weights = np.zeros([self.shape[0], self.shape[1]])

    def forward_propagation(self, X):
        self.input = X
        Z = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(Z)
        return self.output

    def backward(self, delta):
        self.delta = self.activation_derivative(self.output) * delta
        return np.dot(self.delta, self.weights.T)

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * np.dot(self.input.T, self.delta)/100
        self.bias = self.bias - learning_rate * np.sum(self.delta, axis=0)/100
