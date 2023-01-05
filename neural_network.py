from loss_functions import *
from layer import Layer


class NeuralNetwork:
    def __init__(self, input_layer_dim, loss='mse'):
        self.input_dim = input_layer_dim
        self.loss, self.loss_derivative = get_loss_func(loss)
        self.losses = []
        self.layers = []
        self.layer_count = 0

    def add(self, layer: Layer):
        prev_layer = self.input_dim if self.layer_count == 0 else self.layers[self.layer_count - 1].shape[1]
        layer.build(prev_layer)
        self.layers.append(layer)
        self.layer_count += 1

    def desc(self):
        for layer in self.layers:
            print(
                'units:', layer.shape[1], 'Shape:', tuple(layer.shape), 'Activation function:', str(layer.activation),
                'Weight & Bias Initializer:', f'{layer.weight_initializer}, {layer.bias_initializer}'
            )

    def __forward_propagation(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward_propagation(Z)
        return Z

    def __backward_propagation(self, delta, learning_rate):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            layer.update(learning_rate)

    def fit(self, X, y, learning_rate=0.5, epochs=100, print_outputs=True):
        for i in range(epochs):
            y_pred = self.__forward_propagation(X)
            delta = self.loss_derivative(y, y_pred)
            self.__backward_propagation(delta, learning_rate)
            self.losses.append(self.loss(y, y_pred))
            if print_outputs:
                print(f'Epoch: {i} Loss: {self.losses[i]}')
    
    def get_losses(self):
        return self.losses

    def predict(self, X):
        # return [self.forward(sample) for sample in X]
        return self.__forward_propagation(X)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        # acc = sum(1 for i in range(y.shape[0]) if y[i] == y_pred[i]) / len(y) * 100
        return root_mean_squared_error(y, y_pred)
