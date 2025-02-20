import math
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PerceptronLayer import PerceptronLayer
from Activation import Activation, SoftMax, Sigmoid, ReLU
from Loss import Loss, CrossEntropy

class Trainer():
    def __init__(self, features : list, target : list, epochs : int = 8000, learningRate : float = 0.1, error: float = 0.01):
        self.features = np.array(features)
        self.target = np.array(target)
        self.inputSize = self.features.shape[1]
        self.outputs = len(set(target))
        self.targetMap = self.createTargetMap()
        self.trueValues = self.createTrueValues()
        self.convertedTarget = self.convertTarget(target)
        self.epochs = epochs
        self.learningRate = learningRate
        self.targetMap = self.createTargetMap()
        self.convertedTarget = np.array(self.convertTarget(target))
        self.network = self.create_network()
        #optimize_network([perceptronLayer(self.inputs, 1)], 0)

    def create_network(self):
        # come determinare il nunmero corretto di layer e di neuroni per layer?
        hidden_neurons = max(10, int(np.sqrt(self.inputSize)))  # Numero ottimizzato di neuroni nascosti
        network = [
            PerceptronLayer(self.inputSize, hidden_neurons, Sigmoid(), None),  # Nessuna loss qui
            PerceptronLayer(hidden_neurons, self.outputs, SoftMax(), CrossEntropy())  # CrossEntropy solo sull'output
        ]
        return network
     
    def convolutional_dimension(self, features : int, kernel : int = 3, padding : int = 2, stride : int = 1):
        dim = (((math.sqrt(features) - kernel + (2 * padding)) // stride) + 1) ** 2
        return int(dim)

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            network_output = self.features

            for layer in self.network:
                network_output = layer.forward(network_output)


            loss = self.network[-1].get_loss(self.convertedTarget, network_output)
            total_loss += np.sum(loss)


            # Backward pass
            #error = loss
            err = network_output - self.convertedTarget
            error = err
            for layer in reversed(self.network):
                error = layer.backward(error, self.learningRate)

        return self.network

    def convertTarget(self, target):
        encoder = LabelEncoder()
        target_encoded = encoder.fit_transform(target)
        return np.eye(self.outputs)[target_encoded]

    def createTargetMap(self):
        arr = []
        i = 0
        for element in set(self.target):
            arr.append({"key": element, "value" : i})
            i += 1
        return arr
    
    def createTrueValues(self):
        arr = []
        i = 0
        for element in set(self.target):
            index = self.get_target_value(element)
            val = []
            for i in range(self.outputs):
                if(i == index):
                    val.append(1)
                else:
                    val.append(0)
            arr.append(val)
        return arr
    
    def get_target_value(self, search_key):
        for item in self.targetMap:
            if item['key'] == search_key:
                return item['value']
        return None