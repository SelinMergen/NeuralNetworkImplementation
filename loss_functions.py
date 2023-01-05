import numpy as np


def get_loss_func(key):
    loss_func = {
        'mse': [mean_squared_error, mean_squared_error_derivative],
        'rmse': [root_mean_squared_error, root_mean_squared_error_derivative],
        'mae': [mean_absolute_error, mean_absolute_error_derivative],
        'msle': [mean_squared_logarithmic_error, mean_squared_logarithmic_error_derivative],
        'mape': [mean_absolute_percentage_error, mean_absolute_percentage_error_derivative]
    }
    return loss_func[key]


def mean_squared_error(y, y_pred):
    return np.mean((y_pred - y)**2)


def mean_squared_error_derivative(y, y_pred):
    return 2*(y_pred-y)/y.size


def root_mean_squared_error(y, y_pred):
    return np.sqrt(np.mean((y_pred - y)**2))


def root_mean_squared_error_derivative(y, y_pred):
    return (y_pred - y) / np.sqrt(np.mean((y_pred - y)**2))


def mean_absolute_error(y, y_pred):
    return np.mean(np.abs(y_pred - y))


def mean_absolute_error_derivative(y, y_pred):
    return np.sign(y_pred - y)


def mean_squared_logarithmic_error(y, y_pred):
    return np.mean(np.square(np.log(y + 1) - np.log(y_pred + 1)))


def mean_squared_logarithmic_error_derivative(y, y_pred):
    return (np.log(y_pred + 1) - np.log(y + 1)) / (y_pred + 1)


def mean_absolute_percentage_error(y, y_pred):
    return np.mean(np.abs((y_pred - y) / y)) * 100


def mean_absolute_percentage_error_derivative(y, y_pred):
    return (y_pred - y) / (y * (y_pred - y))

