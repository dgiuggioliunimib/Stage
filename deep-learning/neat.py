from typing import List, Optional
import random as rd
from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity

class Node():
    _id_counter = 0
    def __init__(self, bias : float = 1, activation : Activation = Sigmoid()):
        self.bias = bias
        self.id = Node._id_counter
        Node._id_counter += 1 
        self.value = 0
        self.activation_value = 0
        self.activation = activation

    def __repr__(self):
        return f"Node({self.id})"

    def reset(self):
        self.value = 0

    def activate(self, input_value, weight):
        self.value += input_value * weight
        self.activation_value = self.activation.activate(self.value + self.bias)


class Connection():
    def __init__(self, key : Node, value : Node, weight : float):
        self.key = key
        self.value = value
        self.weight = weight

    def __repr__(self):
        return f"{self.key} -> {self.value} (w: {self.weight})"

class Graph():
    def __init__(self, leaves : Optional[List["Node"]] = None):
        self.value = {}
        self.nodes = []
        self.connections = []
        if leaves is not None:
            for leaf in leaves:
                self.add_node(leaf)

    def add_node(self, node):
        if node not in self.nodes:
            self.value[node] = []
            self.nodes.append(node)

    def add_connection(self, connection : Connection):
        self.add_node(connection.key)
        self.add_node(connection.value)
        self.connections.append(connection)
        connection_val = {"value" : connection.value, "weight": connection.weight}
        self.value[connection.key].append(connection_val)
        self.calculate_leaves()

    def remove_connection(self, connection : Connection):
        self.value[connection.key] = [x for x in self.value[connection.key] if x["value"] != connection.value]
        self.connections.remove(connection)

    def calculate_leaves(self):
        leaves = []
        for key in self.value.keys():
            if len(self.value[key]) == 0:
                leaves.append(key)
        self.leaves = leaves

    def get_root(self):
        root = []
        for node in self.nodes:
            is_root = True
            for key in self.value.keys():
                if key != node and is_root:
                    for conn in self.value[key]:
                        if node == conn["value"]:
                            is_root = False
                            break
            if is_root:
                root.append(node)
        return root
    
    def add_root_layer(self, root_layer : List["Node"]):
        previous = self.get_root()
        for r in root_layer:
            for p in previous:
                self.add_connection(Connection(r, p, rd.uniform(-5, 5)))

    def set_rank(self):
        root = self.get_root()
        self.rank = {}
        i = 0
        while len(root) > 0:
            adj = []
            for r in root:
                for v in self.value[r]:
                    val = v["value"]
                    if val not in adj and val not in self.leaves:
                        adj.append(v["value"])
            
            a = adj.copy()
            for node in a:
                w = [x["value"] for x in self.value[node]]
                while len(w) > 0:
                    h = []
                    for w_node in w:
                        if w_node in adj:
                            adj.remove(w_node)
                        h.extend([x["value"] for x in self.value[w_node]])
                    w = h
            self.rank[i] = root
            i += 1
            root = adj
        self.rank[i] = self.leaves
        self.max_rank = i

class Network():
    def __init__(self, graph : Graph = Graph([Node()]), input_size : int = 2):
        self.graph = graph
        self.input_nodes = [Node(bias=0, activation=Identity()) for _ in range(input_size)]
        self.graph.add_root_layer(self.input_nodes)

    def forward(self, input):
        print("input", input)
        self.graph.set_rank()
        output = []
        for i in input:
            rank_zero = self.graph.rank[0]
            for j in range(len(rank_zero)):
                rank_zero[j].activate(i[j], 1)
            for rank in range(self.graph.max_rank):
                for node in self.graph.rank[rank]:
                    for adj in self.graph.value[node]:
                        adj["value"].activate(node.activation_value, adj["weight"])

            single_output = [o.activation_value for o in self.graph.rank[self.graph.max_rank]]
            output.append(single_output)
        return output


class NeatOptimizer():
    def mutate_add_connection(graph: Graph):
        possible_pairs = [(n1, n2) for n1 in graph.nodes for n2 in graph.nodes if n1.id != n2.id]
        if possible_pairs:
            added = False
            while not added:
                in_node, out_node = rd.choice(possible_pairs)
                if GraphAllowConnection().allow_connection(in_node, out_node, graph):
                    connection = Connection(in_node, out_node, rd.uniform(-5, 5))
                    graph.add_connection(connection)
                    added = True

    def mutate_add_node(graph: Graph):
        if graph.connections:
            connection = rd.choice(graph.connections)
            graph.remove_connection(connection)
            new_node = Node()
            graph.add_connection(connection.key, new_node, rd.uniform(-5, 5))
            graph.add_connection(new_node, connection.value, connection.weight, rd.uniform(-5, 5))


class GraphAllowConnection():
    def allow_connection(self, a, b, graph):
        return not(a == b or self.already_exist(a, b, graph) or self.does_make_cycle(a, b, graph))

    def already_exist(self, a, b, graph):
        if a in graph.nodes:
            for adj in graph.value[a]:
                if b == adj["value"]: 
                    return True
        return False

    def does_make_cycle(self, a, b, graph):
        if a == b or b not in graph.nodes:
            return False
        for adj in graph.value[b]:
            if a == adj["value"]:
                return True
        cycle = False
        for adj in graph.value[b]:
            cycle = cycle or self.does_make_cycle(a, adj["value"], graph)
        return cycle


# implementare neat

if __name__ == "__main__":
        

    nodes = []
    gr = Graph()
    #n5 = Node()
    #n6 = Node()
    #nodes = [n1, n2, n3, n4, n5, n6]
    
    #c5 = Connection(n2, n5, rd.uniform(-5, 5))
    #c6 = Connection(n3, n5, rd.uniform(-5, 5))
    #c7 = Connection(n3, n6, rd.uniform(-5, 5))
    #c8 = Connection(n4, n6, rd.uniform(-5, 5))
    #if allow_connection(n1, n2, gr):
        #c1 = Connection(n1, n2, rd.uniform(-5, 5))
        #gr.add_connection(c1)
    #gr.add_connection(c5)
    #gr.add_connection(c6)
    #gr.add_connection(c7)
    #gr.add_connection(c8)


    for i in range(10):
        n = Node(rd.uniform(-5, 5))
        nodes.append(n)
    
    for i in range(50):
        key = rd.choice(nodes)
        val = rd.choice(nodes)
        weight = rd.uniform(-5, 5)
        if GraphAllowConnection().allow_connection(key, val, gr):
            connection = Connection(key, val, weight)
            gr.add_connection(connection)

    net = Network(gr)
    
    #print("net.graph.value", net.graph.value)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    output = net.forward(X)
    print(output)
    print(net.graph.rank)

    NeatOptimizer.mutate_add_connection(net)
    print(net.graph.rank)

# Note:
# La determinazione di quanti input appartengono alla network adesso si fa alla init della classe Network
# ma può essere fatta semplicemente anche alla chiamata della forward (poche righe di codice)

# La classe graph permette di ottenere in input alla creazione una lista di foglie. questo è utile nei casi in cui è
# necessario avere già il numero di foglie corretto per la nostra rete neurale (no random)

# La complessità della forward è elevata perchè la rete non è completa tra ogni layere, quindi dinamicamente
# occorre inoltrare nodo per nodo i valori, rispettando anche il rank che si è determinato

# La forward accetta come input una matrice


# continuare con neat