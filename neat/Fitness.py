from abc import ABC, abstractmethod
import numpy as np
from Network import Network

class Fitness(ABC):
    @abstractmethod
    def calculate(self, network: Network, input, target):
        pass

class AverageError(Fitness):
    def calculate(self, network: Network, input, target):
        output = network.forward(input)
        error = np.abs(np.array(output) - np.array(target))
        fitness = 1 - np.mean(error)
        return fitness
    
class LogarithmSquaredError(Fitness):
    def calculate(self, network: Network, input, target):
        output = network.forward(input)
        squared_error = (np.array(output) - np.array(target)) ** 2
        penalties = -np.log2(1 - np.clip(squared_error, 0, 0.9999) + 1e-9)
        fitness = 1.0 / (1.0 + np.sum(penalties))
        return fitness