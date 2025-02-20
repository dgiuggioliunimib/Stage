import random

class EvolutionaryAlgorithm():

    def __init__(self, population_size = 20, optimization_treshold = 45, mutation_rate = 0.05):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.optimization_treshold = optimization_treshold

    def execute(self, items, generations = 50):
        self.items = items
        self.items_lenght = len(items)
        population = self.generate_population()
    
        for generation in range(generations):
            new_population = []
            
            for _ in range(self.population_size // 2):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            
            population = new_population
            best_solution = max(population, key=self.fitness)
            #print(f"Generazione {generation+1}: Best Fitness = {self.fitness(best_solution)}")

        #return max(population, key=self.fitness)
        print(f"Generazione {generations}: Fitness = {self.fitness(max(population, key=self.fitness))} Solution = {max(population, key=self.fitness)}")

    def generate_population(self):
        return [[random.randint(0, 1) for _ in range(self.items_lenght)] for _ in range(self.population_size)]
    
    def tournament_selection(self, population):
        tournament = random.sample(population, 3)
        return max(tournament, key=self.fitness)
    
    def crossover(self, parent1, parent2):
        point = random.randint(1, self.items_lenght - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    
    def mutate(self, individual):
        for i in range(self.items_lenght):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def fitness(self, individual):
        total_value = sum(self.items[i]["value"] for i in range(self.items_lenght) if individual[i] == 1)
        total_weight = sum(self.items[i]["weight"] for i in range(self.items_lenght) if individual[i] == 1)
        if total_weight > self.optimization_treshold:
            return 0
        return total_value