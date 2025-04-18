from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity

class Node():
    def __init__(self, id: int, activation : Activation = Sigmoid()):
        self.id = id
        self.value = 0
        self.activation_value = 0
        self.activation = activation

    def __eq__(self, value):
        return isinstance(value, Node) and self.id == value.id
    
    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"({self.id})"

    def reset(self):
        self.value = 0

    def activate(self, input_value, weight):
        self.value += input_value * weight
        self.activation_value = self.activation.activate(self.value)