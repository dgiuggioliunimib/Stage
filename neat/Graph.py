from Node import Node
from Connection import Connection
from typing import List, Optional
import random as rd

class Graph():
    def __init__(self, leaves : List[Node] = None):
        self.root = []
        self.connections = []
        if leaves is not None:
            self.set_leaves(leaves)
            self.nodes = leaves.copy()
        else:
            self.nodes = []
            self.leaves = []

    def set_root(self, root):
        self.root = root.copy()

    def set_leaves(self, leaves):
        self.leaves = leaves.copy()

    def get_key_nodes(self):
        return set([c.key for c in self.connections])
    
    def get_values_nodes(self):
        return set([c.value for c in self.connections])
    
    def get_intermediate_nodes(self):
        return set(self.nodes) - set(self.root) - set(self.leaves)
    
    def get_border(self):
        border = self.get_key_nodes() - self.get_values_nodes()
        if len(border) == 0:
            return self.leaves
        else:
            return border
    
    def get_leaves(self):
        return self.get_values_nodes() - self.get_key_nodes() 
    
    def get_nodes(self):
        return self.get_key_nodes() | self.get_values_nodes()
    
    def get_intermediate_connections(self):
        return [c for c in self.connections if c.value not in self.leaves]

    def add_node(self, node: Node):
        if node not in self.nodes:
            #self.value[node] = []
            self.nodes.append(node)

    def remove_node(self, node: Node):
        self.nodes.remove(node)
        for c in self.connections.copy():
            if c.key == node or c.value == node:
                self.remove_connection(c, True)
        #self.calculate_leaves()
        #self.set_rank()
        
    def get_new_id(self):
        max_node = max(self.nodes, key=lambda n: n.id)
        return max_node.id + 1

    def add_connection(self, connection : Connection):
        c = self.get_connection(connection.key, connection.value)
        if not c:
            self.add_node(connection.key)
            self.add_node(connection.value)
            self.connections.append(connection)
        else:
            c.enabled = True

    def remove_connection(self, connection : Connection, delete = False):
        if delete:
            self.connections.remove(connection)
        else:
            connection.enabled = False

    def get_connection(self, key, value):
        for conn in self.connections:
            if conn.key == key and conn.value == value:
                return conn
        return None

    def create_depth_dictionary(self):
        root = self.root
        self.depth_dictionary = {}
        i = 0
        while len(root) > 0:
            adj = set([c.value for c in self.connections if c.enabled and c.key in root and c.value not in self.leaves])
            adj_copy = adj.copy()
            for node in adj_copy:
                w = [c.value for c in self.connections if c.enabled and c.key == node]
                while len(w) > 0:
                    adj = [a for a in adj if a not in w]
                    h = [c.value for c in self.connections if c.enabled and c.key in w]
                    w = h
            self.depth_dictionary[i] = root
            i += 1
            root = adj
        self.depth_dictionary[i] = self.leaves
        self.max_depth = i

    def create_adjacency_list(self):
        self.adjacency_list = {}
        for conn in self.connections:
            if conn.key not in self.adjacency_list:
                self.adjacency_list[conn.key] = []
            self.adjacency_list[conn.key].append({"value": conn.value, "weight": conn.weight, "enabled": conn.enabled})

class GraphAllowConnection():

    def allow_connection(self, a: Node, b: Node, graph : Graph):
        return not(a == b or self.already_exist(a, b, graph) or self.does_make_cycle(a, b, graph))

    def already_exist(self, a: Node, b: Node, graph: Graph):
        connection = graph.get_connection(a, b)
        if connection:
            return True
        else:
            return False
            
    def does_make_cycle(self, a: Node, b: Node, graph: Graph):
        if a == b:
            return True
        if b not in graph.nodes:
            return False
        connection = graph.get_connection(b, a)
        if connection:
            return True
        cycle = False
        for adj in [c.value for c in graph.connections if c.enabled and c.key == b]:
            cycle = cycle or self.does_make_cycle(a, adj, graph)
        return cycle