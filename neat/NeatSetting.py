from Fitness import AverageError
from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity

class NeatSetting:

    def __init__(self, population = 150, 
                 generations = 1000, 
                 error = 0.01, 
                 inherit_disjoint_rate = 0.75, 
                 mutation_add_node_rate = 0.3,
                 mutation_add_connection_rate = 0.1,
                 mutation_change_weight_rate = 0.8,
                 mutatation_perturb_chance = 0.9,
                 mutation_remove_node_rate = 0.005,
                 mutation_remove_connection_rate = 0.005,
                 speciation_threshold = 3.0,
                 fitness = AverageError(),
                 leaves_activation = Sigmoid()):
        
        self.POPULATION = population
        self.GENERATIONS = generations
        self.ERROR = error

        #self.MUTATION_RATE = mutation_rate
        self.INHERIT_DISJOINT_RATE = inherit_disjoint_rate
        self.MUTATION_ADD_NODE_RATE = mutation_add_node_rate
        self.MUTATION_ADD_CONNECTION_RATE = mutation_add_connection_rate
        self.MUTATION_CHANGE_WEIGHT_RATE = mutation_change_weight_rate
        self.MUTATION_PERTURB_CHANCE = mutatation_perturb_chance
        self.MUTATION_REMOVE_NODE_RATE = mutation_remove_node_rate
        self.MUTATION_REMOVE_CONNECTION_RATE = mutation_remove_connection_rate
        #self.MUTATION_REACTIVATE_CONNECTION_RATE = mutation_reactivate_connection_rate

        self.SPECIATION_THRESHOLD = speciation_threshold
        
        self.FITNESS = fitness
        self.LEAVES_ACTIVATION = leaves_activation