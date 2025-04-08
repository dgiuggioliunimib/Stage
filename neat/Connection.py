from Node import Node

class Connection():
    def __init__(self, key : Node, value : Node, weight : float):
        self.key = key
        self.value = value
        self.weight = weight
        self.enabled = True

    def __repr__(self):
        return f"{self.key} -> {self.value} {self.enabled}"