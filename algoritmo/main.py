import random
import math
import matplotlib.pyplot as plt

POPULATION = 5
GENERATIONS = 30
MUTATION_RATE = 0.3
OFFSPRING_COUNT = 5
MIN = -4
MAX = 4

def initialize_population():
    return [random.uniform(MIN, MAX) for _ in range(POPULATION)]

def select(population):
    sorted_population = sorted(population, key=fitness, reverse=True)
    n = int(len(population) / 2)
    return sorted_population[:n]

def fitness(x):
    return math.sin(5 * x) * math.exp(-x/2) + 0.3 * math.cos(2 * x)

def crossover(parent1, parent2):
    return (parent1 + parent2) / 2

def mutate(individual):
    mutated = individual + random.gauss((MIN + MAX) / 2, MAX - MIN)
    return (mutated - MIN) % (MAX - MIN) + MIN


population = initialize_population()
fitness_history = []
for generation in range(GENERATIONS):
    population_string = ""
    
    best_individuals = select(population)
    elite = best_individuals[0]
    offspring = [elite]

    for i in range(len(population)):
        population_string += " [x= " + population[i].__str__() + " f(x)= " + fitness(population[i]).__str__() + "] "
    
    #print(f"Population GENERATION {generation}: {population_string}")
    print(f"GENERATION {generation} Best Individual: x = {elite}, f(x) = {fitness(elite)}")
    
    for _ in range(OFFSPRING_COUNT):
        parent1, parent2 = random.sample(best_individuals, 2)
        child = crossover(parent1, parent2)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        offspring.append(child)
    population = offspring
    best = select(offspring)[0]
    fitness_history.append(fitness(best))

print(f"FINAL BEST INDIVIDUAL: x = {best}, f(x) = {fitness(best)}")
plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Evolution")
plt.show()
    



