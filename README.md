### NeuralNetworkImplementation
You can run the best model with the code given below
```bash
    python3 main.py
```
If you want to initialize a model you can call NeuralNetwork class:
```python
from neural_network import NeuralNetwork
model = NeuralNetwork(input_layer_dim=13, loss='rmse')
```
To do this as you can see you need to give input_layer dimension and loss function as your parameter.

After initializing your model you can add layer with the code given below:
```python
from layer import Layer
model.add(Layer(128, activation='relu', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model.add(Layer(16, activation='sigmoid', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
model.add(Layer(1, activation='linear', bias_initializer='random_uniform', weight_initializer='glorot_uniform'))
```
As layer parameters you can choose the activation function, bias and weight initializers and also you need to give unit count to not to get any error.

If you wish to see the description of your model you can call ``desc()`` method:
```python
model.desc()
```

Finally, with ``fit()`` and ``predict()`` methods you can get predicted values and train your model:
```python
model.fit(X_train, y_train, 0.02, 3000)
y_pred = model.predict(X_test)
```
To get your model accuracy you can call ``accuracy()`` method and see the loss value:
```python
loss_train = model.accuracy(X_train, y_train)
loss_test = model.accuracy(X_test, y_test)
print('RMSE on Train Data:', loss_train)
print('RMSE on Test Data:', loss_test)
```
