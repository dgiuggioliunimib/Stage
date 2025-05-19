from Node import Node
from Connection import Connection, InnovationTracker
from NeatSetting import NeatSetting
from Graph import Graph, GraphAllowConnection
from Network import Network
from typing import List, Optional
import random as rd
import numpy as np
from Fitness import Fitness, AverageError, LogarithmSquaredError
from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity
import matplotlib.pyplot as plt

class Specie():
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.best_fitness = 0
        self.avg_fitness = 0
        self.stagnation_counter = 0
        self.mutation_factor = 1
        self.add_node_factor = 1
        self.weight_influence = 0
        self.bias_influence = 0

    def add_member(self, network):
        self.members.append(network)

class NeatOptimizer():

    innovation_tracker = InnovationTracker()
    gac = GraphAllowConnection()

    def __init__(self, setting : NeatSetting, graph_output_size: int = 1, graph_input_size: int = 2):
        self.setting = setting
        if graph_output_size == 2:
            self.graph_leaves = 1
        else: 
            self.graph_leaves = graph_output_size
        self.graph_input_size = graph_input_size

    def initialize_graph(self):
        return Graph([Node(i + 1, self.setting.LEAVES_ACTIVATION) for i in range(self.graph_leaves)])
    
    def add_root(self, graph : Graph):
        if len(graph.root) == 0:
            bias_node = [Node(0, Identity())]
            input_nodes = [Node(-1 * (i + 1), Identity()) for i in range(self.graph_input_size)]
            root = bias_node
            root.extend(input_nodes)
            graph.set_root(root)

        border = [b for b in graph.get_border() if b.id > 0]
        for r in graph.root:
            for b in border:
                inn = self.innovation_tracker.get_innovation_for_node(r.id, b.id)
                graph.add_connection(Connection(r, b, rd.uniform(-5, 5), inn))

        return graph
    
    def remove_extra_leaves(self, graph: Graph):
        extra_leaves = [l for l in graph.get_leaves() if l.id > self.graph_leaves]
        for e in extra_leaves:
            for l in graph.leaves:
                inn = self.innovation_tracker.get_innovation_for_connection(e.id, l.id)
                graph.add_connection(Connection(e, l, rd.uniform(-5, 5), inn))
        
    def initialize_network(self, graph: Graph = None):
        if graph is None:
            graph = self.initialize_graph()
            self.add_root(graph)

        return Network(graph)

    def optimize(self, input, target):
        population = [self.initialize_network() for _ in range(self.setting.POPULATION)]
        fitness_history = []
        nodes_history = []
        species_list = []
        species_history = []
        population_history = []
        gen = 0
        error = 1
        best_fitness_ever = 0
        best_network_ever = None
        best_gen_ever = 0
        while gen < self.setting.GENERATIONS and error > self.setting.ERROR:
            species_list = self.speciation(population, species_list)
            species_list = sorted(species_list, key=lambda s: s.avg_fitness)
            population = self.evolve_species(species_list, input, target)
            best_network = max(population, key=lambda n: self.setting.FITNESS.calculate(n, input, target))
            best_fitness = self.setting.FITNESS.calculate(best_network, input, target)
            if best_fitness > best_fitness_ever:
                best_network_ever = best_network
                best_fitness_ever = best_fitness
                best_gen_ever = gen + 1
            if (gen+1) % 10 == 0:
                print(f"Generazione {gen+1}, Miglior fitness: {best_fitness:.4f}")

            fitness_history.append(best_fitness)
            nodes_history.append(len(best_network.graph.nodes))
            species_history.append(len(species_list))
            population_history.append(len(population))
            gen += 1
            error = 1 - best_fitness
        
        #print("Best network ever in GEN:", best_gen_ever, "\nFITNESS:", best_fitness_ever, "\n", best_network_ever.graph.rank, "\n", best_network_ever.graph.adjacency_list, "\n", best_network_ever.graph.connections)
        
        '''
        plt.plot(fitness_history)
        plt.xlabel("Generazione")
        plt.ylabel("Fitness")
        plt.title("Evoluzione della Fitness")
        plt.show()

        plt.plot(nodes_history)
        plt.xlabel("Generazione")
        plt.ylabel("Nodes")
        plt.title("Evoluzione numero nodi")
        plt.show()

        plt.plot(species_history)
        plt.xlabel("Generazione")
        plt.ylabel("Specie")
        plt.title("Evoluzione numero specie")
        plt.show()

        plt.plot(population_history)
        plt.xlabel("Generazione")
        plt.ylabel("Popolazione")
        plt.title("Evoluzione popolazione")
        plt.show()

        '''

        return best_network, gen
    
    def evolve_species(self, species_list, input, target):
        new_population = []

        for species in species_list:
            species.members.sort(key=lambda net: self.setting.FITNESS.calculate(net, input, target), reverse=True)
            best_individual_fitness = self.setting.FITNESS.calculate(species.members[0], input, target)
            species_median_fitness = self.setting.FITNESS.calculate(species.members[int(len(species.members) / 2)], input, target)
            species.avg_fitness = species_median_fitness

            if best_individual_fitness > species.best_fitness:
                species.best_fitness = best_individual_fitness
                species.stagnation_counter = 0

                # AGGIUNTE

                '''species.mutation_factor *= rd.uniform(0.75, 0.999)
                species.add_node_factor *= rd.uniform(1.01, 1.05)'''

                # ---
            else:
                species.stagnation_counter += 1

                 # AGGIUNTE

                '''species.mutation_factor *= 1 + rd.uniform(0.01, 0.05) * species.stagnation_counter
                if sum(c.weight for c in species.representative.graph.connections if c.enabled) / len([c for c in species.representative.graph.connections if c.enabled]) > 0:
                    species.weight_influence -= rd.uniform(0.01, 0.025) * species.stagnation_counter
                else:
                    species.weight_influence += rd.uniform(0.01, 0.025) * species.stagnation_counter

                species.add_node_factor *= rd.uniform(1 / len(species.representative.graph.nodes), 1)'''

                     

                # ---

            elites = self.get_elites(species.members, species.avg_fitness, input, target)
                
            if len(elites) < 4:
                elites = species.members[:3]
            new_population.append(elites[0])

            if species.stagnation_counter > 20 and len(new_population) > self.setting.POPULATION * 0.1:
                continue
            
            offspring_count = int(min(max(self.setting.POPULATION * rd.uniform(0.5, 0.75) / len(species_list), int(len(elites) * species.avg_fitness)), self.setting.POPULATION * rd.uniform(1.0, 1.25) / len(species_list)))         
            #offspring_count = len(species.members) - 1
            while offspring_count > 0:
                parent1, parent2 = rd.sample(elites, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child.graph, species.mutation_factor, species.add_node_factor, species.weight_influence)
                new_population.append(child)
                offspring_count -= 1

        return new_population
    
    def get_elites(self, networks: List[Network], avg, input, target):
        elites = []
        for net in networks:
            fitness = self.setting.FITNESS.calculate(net, input, target)
            if fitness > avg:
                elites.append(net)
        return elites
    
    def crossover(self, parent1: Network, parent2: Network):

        child_graph = Graph(parent1.graph.leaves)
        p1_connections = {(c.key, c.value, c.innovation) : c for c in parent1.graph.connections}
        p2_connections = {(c.key, c.value, c.innovation) : c for c in parent2.graph.connections}

        conn = set(p1_connections.keys()).union(set(p2_connections.keys()))
        conn = sorted(conn, key = lambda c: c[1].id)
        conn = sorted(conn, key = lambda c: c[0].id)

        for key in conn:
            if key in p1_connections and key in p2_connections:
                if key[0].id <= 0 or self.gac.allow_connection(key[0], key[1], child_graph):
                    c = Connection(p1_connections[key].key, p1_connections[key].value, p1_connections[key].weight, p1_connections[key].innovation)
                    c.enabled = p1_connections[key].enabled
                    child_graph.add_connection(c)
            elif key in p1_connections:
                if key[0].id <= 0 or (rd.random() < self.setting.INHERIT_DISJOINT_RATE and self.gac.allow_connection(key[0], key[1], child_graph)):
                    c = Connection(p1_connections[key].key, p1_connections[key].value, p1_connections[key].weight, p1_connections[key].innovation)
                    c.enabled = p1_connections[key].enabled
                    child_graph.add_connection(c)

        child_graph.set_root(parent1.graph.root)
        self.remove_extra_leaves(child_graph)
        child = self.initialize_network(self.add_root(child_graph))
        return child
    
    def mutate(self, graph: Graph, mutation_factor: float = 1, add_node_factor: float = 1, weight_influence: float = 0, bias_influence: float = 0):
        if rd.random() < self.setting.MUTATION_ADD_NODE_RATE * add_node_factor:
            self.mutate_add_node(graph)
        if rd.random() < self.setting.MUTATION_ADD_CONNECTION_RATE * mutation_factor:
            self.mutate_add_connection(graph)
        '''if rd.random() < self.setting.MUTATION_REMOVE_NODE_RATE * (2 - add_node_factor):
            self.muatate_remove_node(graph)'''
        '''if rd.random() < self.setting.MUTATION_REMOVE_CONNECTION_RATE * (2 - mutation_factor):
            self.mutate_remove_connection(graph)'''
        for conn in graph.connections:
            '''if not conn.enabled:
                if rd.random() < self.REACTIVATION_CONNECTION_RATE * mutation_factor:
                    conn.enabled = True'''
            if rd.random() < self.setting.MUTATION_CHANGE_WEIGHT_RATE * mutation_factor:
                self.mutate_change_weight(conn, weight_influence)

    def mutate_add_node(self, graph: Graph):
        if graph.connections:
            connection = rd.choice(graph.connections)
            if connection.key.id != 0:
                new_node = Node(graph.get_new_id())
                inn1 = self.innovation_tracker.get_innovation_for_node(new_node.id, connection.value.id)
                graph.add_connection(Connection(new_node, connection.value, connection.weight, inn1))
                inn2 = self.innovation_tracker.get_innovation_for_node(connection.key.id, new_node)
                graph.add_connection(Connection(connection.key, new_node, 1, inn2))
                graph.remove_connection(connection)
            
    def mutate_add_connection(self, graph: Graph):
        possible_pairs = [(n1, n2) for n1 in graph.nodes for n2 in graph.nodes
                          if n1 != n2 and 
                          n1 not in graph.leaves and
                          n2 not in graph.root]
        if possible_pairs:
            added = False
            x = len(possible_pairs)
            while not added and x > 0:
                in_node, out_node = rd.choice(possible_pairs)
                if self.gac.allow_connection(in_node, out_node, graph):
                    inn = self.innovation_tracker.get_innovation_for_connection(in_node.id, out_node.id)
                    connection = Connection(in_node, out_node, rd.uniform(-1, 1), inn)
                    graph.add_connection(connection)
                    added = True
                x -= 1

    def muatate_remove_node(self, graph: Graph):
        intermediate_nodes = [n for n in graph.get_intermediate_nodes()]
        if len(intermediate_nodes) > 0:
            node = rd.choice(intermediate_nodes)
            graph.remove_node(node)
            for r in graph.root:
                has_connections = False
                for c in graph.connections:
                    if c.enabled and c.key == r:
                        has_connections = True
                        break
                if not has_connections:
                    value = rd.choice([n for n in graph.nodes if n.id > 0])
                    inn = self.innovation_tracker.get_innovation_for_connection(r.id, value.id)
                    graph.add_connection(Connection(r, value, rd.uniform(-1, 1)))
            
            for n in [n for n in graph.get_intermediate_nodes()]:
                has_connections_in = False
                has_connections_out = False
                for c in graph.connections:
                    if c.enabled:
                        if c.key == n:
                            has_connections_out = True
                        elif c.value == n:
                            has_connections_in = True
                    if has_connections_in and has_connections_out:
                        break
                
                if not has_connections_in:
                    key = rd.choice([n for n in graph.root])
                    inn = self.innovation_tracker.get_innovation_for_connection(key.id, n.id)
                    graph.add_connection(Connection(key, n, rd.uniform(-1, 1)))
                
                if not has_connections_out:
                    value = rd.choice([n for n in graph.leaves])
                    inn = self.innovation_tracker.get_innovation_for_connection(n.id, value.id)
                    graph.add_connection(Connection(n, value, rd.uniform(-1, 1)))

            for l in graph.leaves:
                has_connections = False
                for c in graph.connections:
                    if c.enabled and c.value == l:
                        has_connections = True
                        break
                if not has_connections:
                    key = rd.choice([n for n in graph.nodes if n.id < 0 or n.id > len(graph.leaves)])
                    inn = self.innovation_tracker.get_innovation_for_connection(key.id, l.id)
                    graph.add_connection(Connection(key, l, rd.uniform(-1, 1), inn))

    def mutate_remove_connection(self, graph: Graph):

        if len(graph.nodes) > len(graph.root) + len(graph.leaves):
            done = False
            i = 0
            while not done and i < len(graph.connections):
                start = rd.choice([n for n in graph.nodes if n not in graph.leaves])
                middle = rd.choice([c.value for c in graph.connections if c.key == start])
                if middle not in graph.leaves:
                    end = rd.choice([c.value for c in graph.connections if c.key == middle])
                    conn1 = graph.get_connection(start, middle)
                    conn2 = graph.get_connection(middle, end)
                    graph.remove_connection(conn1)
                    graph.remove_connection(conn2)
                    inn = self.innovation_tracker.get_innovation_for_connection(start.id, end.id)
                    graph.add_connection(Connection(start, end, rd.uniform(-1, 1), inn))
                    adj = [c.value for c in graph.connections if c.key == middle or c.value == middle]
                    if len(adj) < 1:
                        graph.remove_node(middle)
                    else:
                        if middle in graph.get_border():
                            for r in graph.root:
                                inn = self.innovation_tracker.get_innovation_for_connection(r.id, middle.id)
                                graph.add_connection(Connection(r, middle, rd.uniform(-1, 1), inn))

                        if middle in graph.get_leaves():
                            for l in graph.leaves:
                                inn = self.innovation_tracker.get_innovation_for_connection(middle.id, l.id)
                                graph.add_connection(Connection(middle, l, rd.uniform(-1, 1), inn))
                    done = True
                i += 1

    def mutate_change_weight(self, connection: Connection, weight_influence):
        if rd.random() < self.setting.MUTATION_PERTURB_CHANCE:
            connection.weight += rd.uniform(-0.5 + weight_influence, 0.5 + weight_influence)
        else:
            connection.weight = rd.uniform(-1, 1)

    def calculate_genetic_distance(self, net1: Network, net2: Network, c1=1, c2=1, c3=0.4):

        genes1 = {conn.innovation: conn for conn in net1.graph.connections}
        genes2 = {conn.innovation: conn for conn in net2.graph.connections}

        innov1 = set(genes1.keys())
        innov2 = set(genes2.keys())

        matching = innov1 & innov2
        disjoint = {i for i in (innov1 ^ innov2) if i <= max(innov1.union(innov2))}
        excess = {i for i in (innov1 ^ innov2) if i > min(max(innov1, default=0), max(innov2, default=0))}

        if matching:
            weight_diffs = [
                abs(genes1[i].weight - genes2[i].weight) for i in matching
            ]
            avg_weight_diff = sum(weight_diffs) / len(weight_diffs)
        else:
            avg_weight_diff = 0.0

        N = max(len(genes1), len(genes2))
        if N < 20:
            N = 1

        delta = (c1 * len(excess)) / N + (c2 * len(disjoint)) / N + c3 * avg_weight_diff
        return delta

    def speciation(self, population, species_list):
        species_map = {species: [] for species in species_list}

        for net in population:
            assigned = False

            for species in species_list:
                if self.calculate_genetic_distance(net, species.representative) < self.setting.SPECIATION_THRESHOLD:
                    species_map[species].append(net)
                    assigned = True
                    break 

            if not assigned:
                new_species = Specie(net)
                species_list.append(new_species)
                species_map[new_species] = [net]

        keys = species_map.copy().keys()
        for species in keys:
            if len(species_map[species]) < 4:
                min = 0
                min_sp = None
                for sp in [s for s in species_list if s != species]:
                    gd = self.calculate_genetic_distance(species.representative, sp.representative)
                    if gd < min or min == 0:
                        min = gd
                        min_sp = sp
                if min_sp is not None :
                    for m in species_map[species]:
                        species_map[min_sp].append(m)
                    species_list.remove(species)
                    species_map.pop(species)

        for species in species_list:
            species.members = species_map.get(species, [])

        species_list = [species for species in species_list if species.members]
        return species_list

class TargetTransform():

    def get_true_values(target):
            if len(set(target)) == 2:
                arr = []
                first = target[0]
                for i in range(len(target)):
                    val = []
                    if target[i] == first:
                        val.append(0)
                    else:
                        val.append(1)
                    arr.append(val)
                return arr
            target_map = TargetTransform.create_target_map(target)
            arr = []
            for t in target:
                index = TargetTransform.get_target_value(t, target_map)
                val = []
                for i in range(len(set(target))):
                    if(i == index):
                        val.append(1)
                    else:
                        val.append(0)
                arr.append(val)
            return arr

    def create_target_map(target):
            arr = []
            i = 0
            for element in set(target):
                arr.append({"key": element, "value" : i})
                i += 1
            return arr

    def get_target_value(search_key, target_map):
            for item in target_map:
                if item['key'] == search_key:
                    return item['value']
            return None
