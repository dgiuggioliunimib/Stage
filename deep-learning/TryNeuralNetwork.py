import numpy as np
from Trainer import Trainer
from PerceptronLayer import PerceptronLayer

class TryNeuralNetwork():
    def __init__(self):
        self.network = []

    def train(self, features, target):
        trainer = Trainer(features, target)
        self.network = trainer.train()
        layer = 0
        for pl in self.network:
            layer += 1
            print("Layer:", layer, "Neurons:", pl.neurons, "Inputs", pl.inputSize)
    
    def predict(self, inputs):
        output = inputs
        for layer in self.network:
            output = layer.forward(output)
        print(output)
        predicted = np.argmax(output, axis=1)
        return predicted