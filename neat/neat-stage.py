from Node import Node
from Connection import Connection
from Graph import Graph
from Network import Network
from typing import List, Optional
import random as rd
import numpy as np
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

    #MUTATION_RATE = 0.05
    INHERIT_DISJOINT_RATE = 0.75
    MUTATION_ADD_NODE_RATE = 0.01
    MUTATION_ADD_CONNECTION_RATE = 0.02
    MUTATION_REMOVE_NODE_RATE = 0.005
    MUTATION_REMOVE_CONNECTION_RATE = 0.005
    MUTATION_CHANGE_WEIGHT_RATE = 0.01
    REACTIVATION_CONNECTION_RATE = 0.005
    MUTATION_CHANGE_BIAS_RATE = 0.005

    def __init__(self, population: int, generations: int, graph_leaves: int = 1, graph_input_size: int = 2, error: float = 0.03):
        self.POPULATION = population
        self.GENERATIONS = generations
        self.graph_leaves = graph_leaves
        self.graph_input_size = graph_input_size
        self.ERROR = error

    def initialize_graph(self, leaves_activation: Activation = Sigmoid()):
        return Graph([Node(i + 1, rd.random(), leaves_activation) for i in range(self.graph_leaves)])
    
    def add_root(self, graph : Graph):
        
        if len(graph.root) == 0:
            input_nodes = [Node(-1 * (i + 1), 0, Identity()) for i in range(self.graph_input_size)]
            graph.set_root(input_nodes)

        border = [b for b in graph.get_border() if b.id > 0]
        for r in graph.root:
            for b in border:
                graph.add_connection(Connection(r, b, rd.uniform(-2.5, 2.5)))

        return graph
        
    def initialize_network(self, graph: Graph = None, leaves_activation: Activation = Sigmoid()):
        if graph is None:
            graph = self.initialize_graph(leaves_activation)
            self.add_root(graph)

        return Network(graph)

    def optimize(self, input, target):
        population = [optimizer.initialize_network() for _ in range(self.POPULATION)]
        fitness_history = []
        nodes_history = []
        species_list = []
        species_history = []
        gen = 0
        error = 1
        best_fitness_ever = 0
        best_net_ever = None
        best_gen_ever = 0
        while gen < self.GENERATIONS and error > self.ERROR:
            species_list = self.speciation(population, species_list)
            # COSTOSO!!!
            #species_list = sorted(species_list, key=lambda s: max(self.fitness(s.members[i], input, target) for i in range(len(s.members))))
            species_list = sorted(species_list, key=lambda s: s.avg_fitness)

            #population = self.evolve(population, input, target)
            population = self.evolve_species(species_list, input, target)
            #print("pop len", len(population))
            best_network = max(population, key=lambda n: self.fitness(n, input, target))
            best_fitness = self.fitness(best_network, input, target)
            if best_fitness > best_fitness_ever:
                best_network_ever = best_network
                best_fitness_ever = best_fitness
                best_gen_ever = gen + 1
            nodes = len(best_network.graph.nodes)
            #print("species", len(species_list), "population:", len(population))
            if (gen+1) % 1 == 0:
                print(f"Generazione {gen+1}, Miglior fitness: {best_fitness:.4f} \nBest Network: {best_network.graph.rank} \nPopulation: {len(population)} ({len(species_list)} species)")
                #print(f"Output: {best_network.forward(input)}")
            fitness_history.append(best_fitness)
            nodes_history.append(nodes)
            species_history.append(len(species_list))
            gen += 1
            error = 1 - best_fitness
        
        print("Best network ever in GEN:", best_gen_ever, "\nFITNESS:", best_fitness_ever, "\n", best_network_ever.graph.rank)
        
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
        return best_network
    
    def evolve(self, population: List[Network], input, target):
        parents = self.selection(population, input, target)
        children = []
        while len(children) < len(population) - len(parents):
            parent1, parent2 = rd.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            if rd.random() < self.MUTATION_RATE:
                self.mutate(child.graph)
            children.append(child)

        parents.extend(children)
        return parents
    
    def evolve_species(self, species_list, input, target):
        new_population = []
        allow_normal_crossover = False

        for species in species_list:
            species.members.sort(key=lambda net: self.fitness(net, input, target), reverse=True)
            best_individual_fitness = self.fitness(species.members[0], input, target)
            species_median_fitness = self.fitness(species.members[int(len(species.members) / 2)], input, target)
            species.avg_fitness = species_median_fitness

            # and species.avg_fitness > species_median_fitness
            if best_individual_fitness > species.best_fitness:
                species.best_fitness = best_individual_fitness
                species.stagnation_counter = 0
                allow_normal_crossover = True

                # AGGIUNTE

                species.mutation_factor *= rd.uniform(0.75, 0.999)
                species.add_node_factor *= rd.uniform(1.01, 1.05)

                # ---

            else:
                species.stagnation_counter += 1

                # AGGIUNTE

                species.mutation_factor *= 1 + rd.uniform(0.01, 0.05) * species.stagnation_counter
                if sum(c.weight for c in species.representative.graph.connections if c.enabled) / len([c for c in species.representative.graph.connections if c.enabled]) > 0:
                    species.weight_influence -= rd.uniform(0.01, 0.025) * species.stagnation_counter
                else:
                    species.weight_influence += rd.uniform(0.01, 0.025) * species.stagnation_counter

                if sum(n.bias for n in species.representative.graph.nodes) / len(species.representative.graph.nodes) > 0:
                    species.bias_influence -= rd.uniform(0.001, 0.0025) * species.stagnation_counter
                else:
                     species.bias_influence += rd.uniform(0.001, 0.0025) * species.stagnation_counter
                species.add_node_factor *= rd.uniform(1 / len(species.representative.graph.nodes), 1)

                # ---

            if species.stagnation_counter > 20 and len(new_population) > self.POPULATION * 0.10:
                continue
                #allow_normal_crossover = True

            elites = self.get_elites(species.members, species.avg_fitness, input, target)
                
            new_population.extend(elites)
            if len(elites) < 4:
                elites = species.members[:3]
            
            offspring_count = max(self.POPULATION * 0.15 , int(len(elites) * species.avg_fitness * (rd.randint(75, 200) / 100)))
            offspring_count = min(self.POPULATION / len(species_list), offspring_count)

            #offspring_count = int(n - n / len(elites))
            #print("offspring count", offspring_count)
            #else:
                #offspring_count = len(elites)

            #print(species_avg_fitness, total_average_fitness, "offspring count", offspring_count)

            while offspring_count > 0:
                parent1, parent2 = rd.sample(elites, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child.graph, species.mutation_factor, species.add_node_factor, species.weight_influence, species.bias_influence)
                new_population.append(child)
                offspring_count -= 1
            
        '''if allow_normal_crossover:
            species = species_list
        else:
            species = species_list[:2]'''

        return new_population
    
    def get_elites(self, networks: List[Network], avg, input, target):
        elites = []
        for net in networks:
            fitness = self.fitness(net, input, target)
            if fitness > avg:
                elites.append(net)
        return elites

    def selection(self, population: List[Network], input, target):
        sorted_population = sorted(population, key=lambda n: self.fitness(n, input, target), reverse=True)
        return sorted_population[:len(population) // 2]
    
    def fitness(self, network: Network, input, target):
        output = network.forward(input)
        error = np.mean(np.abs(np.array(output) - np.array(target)))
        return error

        #fitness_value = -error
        #normalized_fitness = np.exp(fitness_value)
        #return normalized_fitness


        #error = 0
        #for i in range(len(output)):
            #for j in range(len(output[i])):
                #error -= abs(output[i][j] - target[i][j])
        #return error
    
    def crossover(self, parent1: Network, parent2: Network):

        #child_graph = self.initialize_graph()
        child_graph = Graph(parent1.graph.leaves)
        p1_connections = {(c.key, c.value) : c for c in parent1.graph.connections}
        p2_connections = {(c.key, c.value) : c for c in parent2.graph.connections}

        #common = set(p1_connections.keys()).intersection(set(p2_connections.keys()))
        #not_common = set(p1_connections.keys()).union(set(p2_connections.keys())) - common
        conn = set(p1_connections.keys()).union(set(p2_connections.keys()))
        conn = sorted(conn, key = lambda c: c[1].id)
        conn = sorted(conn, key = lambda c: c[0].id)
        gac = GraphAllowConnection()
        #parent = rd.choice([parent1, parent2])
        #child_graph.set_root(parent.graph.root)
        #child_graph.set_leaves(parent.graph.leaves)

        '''for key in common:
            child_graph.add_connection(Connection(p1_connections[key].key, p1_connections[key].value, p1_connections[key].weight))
        '''
        for key in conn:
            if key in p1_connections and key in p2_connections:
                if key[0].id < 0 or gac.allow_connection(key[0], key[1], child_graph):
                    child_graph.add_connection(Connection(p1_connections[key].key, p1_connections[key].value, p1_connections[key].weight))
                    
            elif key in p1_connections:
                if key[0].id < 0 or (rd.random() < self.INHERIT_DISJOINT_RATE and gac.allow_connection(key[0], key[1], child_graph)):
                    child_graph.add_connection(Connection(p1_connections[key].key, p1_connections[key].value, p1_connections[key].weight))
        
        extra_leaves = [l for l in child_graph.get_leaves() if l.id > self.graph_leaves]
        for e in extra_leaves:
            for l in child_graph.leaves:
                child_graph.add_connection(Connection(e, l, rd.uniform(-1, 1)))
        child_graph.set_root(parent1.graph.root)
        child = self.initialize_network(self.add_root(child_graph))
        return child
    
    def mutate(self, graph: Graph, mutation_factor: float = 1, add_node_factor: float = 1, weight_influence: float = 0, bias_influence: float = 0):
        if rd.random() < self.MUTATION_ADD_NODE_RATE * add_node_factor:
            self.mutate_add_node(graph)
        if rd.random() < self.MUTATION_ADD_CONNECTION_RATE * mutation_factor:
            self.mutate_add_connection(graph)
        if rd.random() < self.MUTATION_REMOVE_NODE_RATE * (2 - add_node_factor):
            self.muatate_remove_node(graph)
        if rd.random() < self.MUTATION_REMOVE_CONNECTION_RATE * (2 - mutation_factor):
            self.mutate_remove_connection(graph)
        for node in graph.nodes:
            if rd.random() < self.MUTATION_CHANGE_BIAS_RATE:
                self.mutate_change_bias(node, bias_influence)
        for conn in graph.connections:
            '''if not conn.enabled:
                if rd.random() < self.REACTIVATION_CONNECTION_RATE * mutation_factor:
                    conn.enabled = True'''
            if rd.random() < self.MUTATION_CHANGE_WEIGHT_RATE * mutation_factor:
                self.mutate_change_weight(conn, weight_influence)

    def mutate_add_node(self, graph: Graph):
        if graph.connections:
            connection = rd.choice(graph.connections)
            new_node = Node(graph.get_new_id(), rd.uniform(-1, 1))
            #print("new node", new_node)
            #print("nodes before", graph.nodes)
            graph.add_connection(Connection(new_node, connection.value, rd.uniform(-1, 1)))
            graph.add_connection(Connection(connection.key, new_node, rd.uniform(-1, 1)))
            if rd.random() < 0.5:
                graph.remove_connection(connection)
            #print("nodes after", graph.nodes)
            #print("graph now", graph.connections)
            
    def mutate_add_connection(self, graph: Graph):
        possible_pairs = [(n1, n2) for n1 in graph.nodes for n2 in graph.nodes
                          if n1 != n2 and 
                          n1 not in graph.leaves and
                          n2 not in graph.root]
        '''for c in graph.connections:
            for p in possible_pairs.copy():
                if c.enabled and p[0] == c.key and p[1] == c.value:
                    possible_pairs.remove(p)'''
        if possible_pairs:
            #print("possible pairs", possible_pairs)
            added = False
            x = 10
            gac = GraphAllowConnection()
            while not added and x > 0:
                in_node, out_node = rd.choice(possible_pairs)
                if gac.allow_connection(in_node, out_node, graph):
                    connection = Connection(in_node, out_node, rd.uniform(-1, 1))
                    graph.add_connection(connection)
                    added = True
                x -= 1

    def mutate_change_bias(self, node: Node, bias_influence: float):
        node.bias += rd.uniform(-0.25, 0.25) + bias_influence

    def muatate_remove_node(self, graph: Graph):
        intermediate_nodes = [n for n in graph.get_intermediate_nodes()]
        if len(intermediate_nodes) > 0:
            node = rd.choice(intermediate_nodes)
            #print("node removed", node)
            graph.remove_node(node)
            for r in graph.root:
                has_connections = False
                for c in graph.connections:
                    if c.enabled and c.key == r:
                        has_connections = True
                        break
                if not has_connections:
                    value = rd.choice([n for n in graph.nodes if n.id > 0])
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
                    graph.add_connection(Connection(key, n, rd.uniform(-1, 1)))
                
                if not has_connections_out:
                    value = rd.choice([n for n in graph.leaves])
                    graph.add_connection(Connection(n, value, rd.uniform(-1, 1)))

            for l in graph.leaves:
                has_connections = False
                for c in graph.connections:
                    if c.enabled and c.value == l:
                        has_connections = True
                        break
                if not has_connections:
                    key = rd.choice([n for n in graph.nodes if n.id < 0 or n.id > len(graph.leaves)])
                    graph.add_connection(Connection(key, l, rd.uniform(-1, 1)))

    def mutate_remove_connection(self, graph: Graph):

        if len(graph.nodes) > len(graph.root) + len(graph.leaves):
            #print(graph.connections)
            done = False
            i = 0
            root = graph.root
            while not done and i < len(graph.connections):
                start = rd.choice([n for n in graph.nodes if n not in graph.leaves])
                middle = rd.choice([c.value for c in graph.connections if c.key == start])
                if middle not in graph.leaves:
                    end = rd.choice([c.value for c in graph.connections if c.key == middle])
                    conn1 = graph.get_connection(start, middle)
                    conn2 = graph.get_connection(middle, end)
                    graph.remove_connection(conn1)
                    graph.remove_connection(conn2)
                    graph.add_connection(Connection(start, end, rd.uniform(-1, 1)))
                    adj = [c.value for c in graph.connections if c.key == middle or c.value == middle]
                    if len(adj) < 1:
                        graph.remove_node(middle)
                    else:
                        if middle in graph.get_border():
                            for r in graph.root:
                                graph.add_connection(Connection(r, middle, rd.uniform(-1, 1)))

                        if middle in graph.get_leaves():
                            for l in graph.leaves:
                                graph.add_connection(Connection(middle, l, rd.uniform(-1, 1)))
                    done = True
                i += 1
            #print(graph.connections)

    def mutate_change_weight(self, connection: Connection, weight_influence):
        connection.weight += rd.uniform(-0.25 + weight_influence, 0.25 + weight_influence)

    def calculate_genetic_distance(self, net1: Network, net2: Network, c1=1, c2=1, c3=1):
        nodes1 = set(net1.graph.nodes)
        nodes2 = set(net2.graph.nodes)

        # Determina connessioni in eccesso e disgiunte
        excess_genes = len(nodes1 - nodes2) + len(nodes2 - nodes1)
        disjoint_genes = len((nodes1 | nodes2) - (nodes1 & nodes2))

        # Calcola la differenza media dei pesi
        #common_nodes = nodes1 & nodes2

        #nodes1, connections1 = self.calculate_network_dimension(net1)
        #nodes2, connections2 = self.calculate_network_dimension(net2)
        weight_diff = 0
        for conn1 in net1.graph.connections:
            conn2 = net2.graph.get_connection(conn1.key, conn1.value)
            if conn2:
                #print(conn1.weight, conn2.weight)
                weight_diff += abs(conn1.weight - conn2.weight)
                    
        return c1 * excess_genes + c2 * disjoint_genes + c3 * weight_diff
        '''# Normalizzazione con il numero massimo di connessioni
        N = max(len(nodes1), len(nodes2), 1)  
        genetic_distance = (c1 * excess_genes / N) + (c2 * disjoint_genes / N) + (c3 * weight_diff)
        #print("weight diff", weight_diff, "excess genes", excess_genes, "disjoint genes", disjoint_genes, "genetic distance", genetic_distance)
        return genetic_distance'''
    
    def calculate_network_dimension(self, network: Network):
        nodes = len(network.graph.nodes)
        connections = len([c for c in network.graph.connections if c.enabled])
        return nodes, connections

    def speciation(self, population, species_list, speciation_threshold=4):
        species_map = {species: [] for species in species_list}

        for net in population:
            assigned = False

            for species in species_list:
                if self.calculate_genetic_distance(net, species.representative) < speciation_threshold:
                    species_map[species].append(net)
                    assigned = True
                    break 

            if not assigned:
                new_species = Specie(net)
                species_list.append(new_species)
                species_map[new_species] = [net]

        keys = species_map.copy().keys()
        for species in keys:
            if len(species_map[species]) < self.POPULATION * 0.08:
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
        
class GraphAllowConnection():

    def allow_connection(self, a: Node, b: Node, graph : Graph):
        return not(a == b or self.already_exist(a, b, graph) or self.does_make_cycle(a, b, graph))

    def already_exist(self, a: Node, b: Node, graph: Graph):
        connection = graph.get_connection(a, b)
        if connection:
            return connection.enabled
        else:
            return False
            
    def does_make_cycle(self, a: Node, b: Node, graph: Graph):

        if a == b:
            return True
        if b not in graph.nodes:
            return False
        connection = graph.get_connection(b, a)
        if connection:
            if connection.enabled:
                return True
        cycle = False
        for adj in [c.value for c in graph.connections if c.enabled and c.key == b]:
            cycle = cycle or self.does_make_cycle(a, adj, graph)
        return cycle

class TargetTransform():

    def get_true_values(target):
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


if __name__ == "__main__":

    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    import numpy as np

    breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets 

    X.iloc[:, 5] = X.iloc[:, 5].replace([None, np.nan], 0)

    X = X.to_numpy()
    y = y.to_numpy()

    y = y.flatten()
    '''
    
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    y_true_values = TargetTransform.get_true_values(y)

    '''
    y_true_values = TargetTransform.get_true_values(y)
    test_size = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y_true_values, test_size= test_size, random_state=100)
    print("train set: ", X_train)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    LEAVES = len(set(y))
    INPUT_SIZE = len(X[0])

    POPULATION = 100
    GENERATIONS = 1000

    optimizer = NeatOptimizer(POPULATION, GENERATIONS, LEAVES, INPUT_SIZE)
    network = optimizer.optimize(X_train, y_train)

    for i in range(len(X_test)):
        fitness = optimizer.fitness(network, [X_test[i]], y_test[i])
        print(f"FITNESS Best Network on test input[{i}]: {fitness}")
    # X, y_true_values



# Note:
# La determinazione di quanti input appartengono alla network adesso si fa alla init della classe Network
# ma può essere fatta semplicemente anche alla chiamata della forward (poche righe di codice)

# La classe graph permette di ottenere in input alla creazione una lista di foglie. questo è utile nei casi in cui è
# necessario avere già il numero di foglie corretto per la nostra rete neurale (no random)

# La complessità della forward è elevata perchè la rete non è completa tra ogni layer, quindi dinamicamente
# occorre inoltrare nodo per nodo i valori, rispettando anche il rank che si è determinato

# La forward accetta come input una matrice
