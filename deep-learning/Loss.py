from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    def __init__(self):
        self.eps = 1e-15

    @abstractmethod
    def calculate(self, true, predicted):
        pass

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        
    def calculate(self, true, predicted):
        return -np.sum(true * np.log(predicted + self.eps)) / true.shape[0]
    
    def derivative(self, true, predicted):
        return -np.divide(true, (predicted + self.eps))