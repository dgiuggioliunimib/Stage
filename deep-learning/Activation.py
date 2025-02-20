from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def activate(self, input):
        pass

    @abstractmethod
    def derivative(self, input):
        pass

class Sigmoid(Activation):
    def activate(self, input):
        return 1 / (1 + np.exp(-input))
    
    def derivative(self, input):
        sig = self.activate(input)
        return sig * (1 - sig)
    
class SoftMax(Activation):
    def activate(self, input):
        exp_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
       
    def derivative(self, input):
        softmax_out = self.activate(input)
        return softmax_out * (1 - softmax_out)
    
class ReLU(Activation):
    def activate(self, input):
        return np.maximum(0, input)
    
    def derivative(self, input):
        return (input > 0).astype(float)