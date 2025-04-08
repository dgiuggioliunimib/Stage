from Node import Node
from Connection import Connection
from Graph import Graph
import random as rd
from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity

class Network():

    def __init__(self, graph : Graph = Graph()):
        self.graph = graph

    def forward(self, input):
        self.graph.set_rank()
        self.graph.create_adjacency_list()
        #print("conn", self.graph.connections)
        #print(self.graph.rank)
        #print(self.graph.adjacency_list)
        output = []

        for i in input:
            rank_zero = self.graph.rank[0]
            for j in range(len(rank_zero)):
                rank_zero[j].activate(i[j], 1)
            for rank in range(self.graph.max_rank):
                for node in self.graph.rank[rank]:
                    for adj in self.graph.adjacency_list[node]:
                        adj["value"].activate(node.activation_value, adj["weight"])

            single_output = [o.activation_value for o in self.graph.rank[self.graph.max_rank]]
            output.append(single_output)

            for n in self.graph.nodes:
                n.reset()
        return output
