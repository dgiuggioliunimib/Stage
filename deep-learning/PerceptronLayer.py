import numpy as np
from Activation import Activation
from Loss import Loss

class PerceptronLayer():
    def __init__(self, n_inputs, n_neurons, activation : Activation, loss : Loss):
        self.inputSize = n_inputs
        self.neurons = n_neurons
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(1, n_neurons)
        self.activation = activation
        self.loss = loss

    def forward(self, inputs):
        self.input_data = np.atleast_2d(inputs)
        self.z = np.dot(self.input_data, self.weights) + self.biases
        return self.activation.activate(self.z)
    
    def get_loss(self, true, predicted):
        return self.loss.derivative(true, predicted)
    
    def backward(self, error, learning_rate = 0.1):
        activation_grad = self.activation.derivative(self.z)
        delta = error * activation_grad
        self.update(delta, learning_rate, len(self.input_data))
        return np.dot(delta, self.weights.T)

    def update(self, delta, learning_rate, n):
        d_W = np.dot(self.input_data.T, delta) / n
        d_b = np.mean(delta)

        self.weights -= learning_rate * d_W
        self.biases -= learning_rate * d_b