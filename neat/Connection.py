from Node import Node

class Connection():
    def __init__(self, key : Node, value : Node, weight : float, innovation: int = 0):
        self.key = key
        self.value = value
        self.weight = weight
        self.enabled = True
        self.innovation = innovation

    def __eq__(self, value):
        return isinstance(value, Connection) and (self.key == value.key and 
                                                  self.value == value.value and 
                                                  self.innovation == value.innovation)
    
    def __hash__(self):
        return hash(self.key + " " + self.value + " " + self.innovation)

    def __repr__(self):
        return f"{self.key} -> {self.value} {self.enabled}"
    
class InnovationTracker:
    def __init__(self):
        self.current_innovation = 0
        self.innovations = {}

    def get_innovation_for_connection(self, in_node, out_node):
        key = ("conn", in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.innovations[key]

    def get_innovation_for_node(self, in_node, out_node):
        key = ("node", in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.innovations[key]