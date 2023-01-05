import numpy as np
import pandas as pd
from helper_functions import min_max_scaler, train_test_split
from neural_network import NeuralNetwork
from layer import Layer

np.random.seed(12345)
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)

y = df.MEDV
X = df.drop(['MEDV'], axis=1)

X = min_max_scaler(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = NeuralNetwork(13, 'rmse')
model.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model.add(Layer(16, activation='sigmoid', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model.desc()

model.fit(X_train, y_train, 0.02, 3000)

loss_train = model.accuracy(X_train, y_train)
loss_test = model.accuracy(X_test, y_test)
print('RMSE on Train Data:', loss_train)
print('RMSE on Test Data:', loss_test)

"""
# In here you can see how i draw the graph on report.
# Epoch
epoch_counts = [1000, 3000, 4000, 5000, 7000, 10000, 15000]
rmse_on_epoch_train = []
rmse_on_epoch_test = []

for epoch in epoch_counts:
    np.random.seed(12345)
    model = NeuralNetwork(13, 'rmse')
    model.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
    model.add(Layer(16, activation='sigmoid', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
    model.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
    model.fit(X_train, y_train, 0.02, epoch, False)
    print(f'Epoch:{epoch}')
    loss_train = model.accuracy(X_train, y_train)
    loss_test = model.accuracy(X_test, y_test)
    rmse_on_epoch_train.append(loss_train)
    rmse_on_epoch_test.append(loss_test)
    print(f'loss_train: {loss_train}, loss_test: {loss_test}')
    
# Learning rate
learning_rates = [0.1, 0.05, 0.02, 0.01]
rmse_on_learning_rates_train = []
rmse_on_learning_rates_test = []

for learning_rate in learning_rates:
    np.random.seed(12345)
    model = NeuralNetwork(13, 'rmse')
    model.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
    model.add(Layer(16, activation='sigmoid', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
    model.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
    model.fit(X_train, y_train, learning_rate, 3000, False)
    print(f'Learning rate:{learning_rate}')
    loss_train = model.accuracy(X_train, y_train)
    loss_test = model.accuracy(X_test, y_test)
    rmse_on_learning_rates_train.append(loss_train)
    rmse_on_learning_rates_test.append(loss_test)
    print(f'loss_train: {loss_train}, loss_test: {loss_test}')
    
# layer count
np.random.seed(12345)

model2 = NeuralNetwork(13, 'rmse')
model2.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model2.add(Layer(16, activation='sigmoid', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model2.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model2.fit(X_train, y_train, 0.02, 3000, False)

model1 = NeuralNetwork(13, 'rmse')
model1.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model1.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model1.fit(X_train, y_train, 0.02, 3000, False)

model0 = NeuralNetwork(13, 'rmse')
model0.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model0.fit(X_train, y_train, 0.02, 3000, False)

#unit count
np.random.seed(12345)

model0 = NeuralNetwork(13, 'rmse')
model0.add(Layer(256, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model0.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model0.fit(X_train, y_train, 0.02, 3000, False)

model1 = NeuralNetwork(13, 'rmse')
model1.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model1.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model1.fit(X_train, y_train, 0.02, 3000, False)

model2 = NeuralNetwork(13, 'rmse')
model2.add(Layer(64, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model2.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model2.fit(X_train, y_train, 0.02, 3000, False)

model3 = NeuralNetwork(13, 'rmse')
model3.add(Layer(32, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model3.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model3.fit(X_train, y_train, 0.02, 3000, False)

model4 = NeuralNetwork(13, 'rmse')
model4.add(Layer(16, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model4.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model4.fit(X_train, y_train, 0.02, 3000, False)

"""





