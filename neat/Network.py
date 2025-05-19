from Node import Node
from Connection import Connection
from Graph import Graph
import random as rd
from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity

class Network():

    def __init__(self, graph : Graph = Graph()):
        self.graph = graph

    def forward(self, input):
        self.graph.create_depth_dictionary()
        self.graph.create_adjacency_list()
        output = []

        for i in input:
            rank_zero = self.graph.depth_dictionary[0]
            rank_zero = sorted(rank_zero, key=lambda z: z.id, reverse=True)
            rank_zero[0].activate(1, 1)
            for j in range(1, len(rank_zero)):
                    rank_zero[j].activate(i[j - 1], 1)
            for rank in range(self.graph.max_depth):
                for node in self.graph.depth_dictionary[rank]:
                    for adj in self.graph.adjacency_list[node]:
                        if adj["enabled"]:
                            adj["value"].activate(node.activation_value, adj["weight"])

            single_output = [o.activation_value for o in self.graph.depth_dictionary[self.graph.max_depth]]
            output.append(single_output)

            for n in self.graph.nodes:
                n.reset()
        return output
