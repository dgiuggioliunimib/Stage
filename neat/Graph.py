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
            #connection_val = {"value" : connection.value}
            #self.value[connection.key].append(connection_val)
        else:
            c.enabled = True
 
        #self.calculate_leaves()
        #self.set_rank()

    def remove_connection(self, connection : Connection, delete = False):
        if delete:
            self.connections.remove(connection)
        else:
            connection.enabled = False
        #self.calculate_leaves()
        #self.set_rank()

    def get_connection(self, key, value):
        for conn in self.connections:
            if conn.key == key and conn.value == value:
                return conn
        return None

    '''def calculate_leaves(self):
        leaves = []
        for key in self.value.keys():
            if len(self.value[key]) == 0:
                leaves.append(key)
        self.leaves = leaves'''

    '''def get_leaves(self):
        return self.leaves'''

    '''def get_root(self):
        root = []
        for node in self.nodes:
            is_root = True
            for key in self.value.keys():
                if key != node and is_root:
                    for conn in self.value[key]:
                        value = conn["value"]
                        if node == value:
                            connection = self.get_connection(key, value)
                            if connection.enabled:
                                is_root = False
                                break
            if is_root:
                root.append(node)
        return root'''
    
    '''def add_root_layer(self, root_layer : List[Node]):
        previous = self.get_root()
        for r in root_layer:
            for p in previous:
                self.add_connection(Connection(r, p, rd.uniform(-2.5, 2.5)))'''

    '''def get_intermediate_connections(self):
        connections = self.connections.copy()
        root = self.get_root()
        for c in self.connections:
            if c.key in root:
                connections.remove(c)
        return connections'''

    '''def set_rank(self):
        root = self.get_root()
        self.rank = {}
        i = 0
        while len(root) > 0:
            adj = []
            for r in root:
                for v in self.value[r]:
                    val = v["value"]
                    connection = self.get_connection(r, val)
                    if connection.enabled and val.id not in [a.id for a in adj] and val.id not in [l.id for l in self.leaves]:
                        adj.append(val)
            a = adj.copy()
            for node in a:
                w = [x["value"] for x in self.value[node] if self.get_connection(node, x["value"]).enabled]
                while len(w) > 0:
                    h = []
                    for w_node in w:
                        if w_node in adj:
                            adj.remove(w_node)
                        h.extend([x["value"] for x in self.value[w_node] if self.get_connection(w_node, x["value"]).enabled])
                    w = h
            self.rank[i] = root
            i += 1
            root = adj
        self.rank[i] = self.leaves
        self.max_rank = i'''

    def set_rank(self):
        root = self.root
        self.rank = {}
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
            self.rank[i] = root
            i += 1
            root = adj
        self.rank[i] = self.leaves
        self.max_rank = i

    def create_adjacency_list(self):
        self.adjacency_list = {}
        for conn in self.connections:
            if conn.key not in self.adjacency_list:
                self.adjacency_list[conn.key] = []
            self.adjacency_list[conn.key].append({"value": conn.value, "weight": conn.weight})
