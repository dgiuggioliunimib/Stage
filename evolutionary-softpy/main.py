from candidate import BitVectorCandidate  
from genetic import GeneticAlgorithm
from selection import tournament_selection, stochastic_universal_selection, fitness_prop_selection
import random
import timeit

# Parametri
NUM_ITEMS = 10
POP_SIZE = 45
MUTATION_RATE = 0.05
NUM_GENERATIONS = 20
OPTIMIZATION_TRESHOLD = 50

items = [{ "weight": random.randint(1, 50), "value": random.randint(1, 100)} for _ in range(NUM_ITEMS)]
#print(items)

def fitness(individual):
        print("main 18", individual.candidate)
        if len(individual.candidate) != NUM_ITEMS:
            raise ValueError(f"Invalid candidate length: expected {NUM_ITEMS}, got {len(individual.candidate)}")
        total_value = sum(items[i]["value"] for i in range(NUM_ITEMS) if individual.candidate[i] == 1)
        total_weight = sum(items[i]["weight"] for i in range(NUM_ITEMS) if individual.candidate[i] == 1)
        if total_weight > OPTIMIZATION_TRESHOLD:
            return 0
        return total_value

ga = GeneticAlgorithm(
        candidate_type=BitVectorCandidate,
        fitness_func=fitness,
        pop_size=POP_SIZE,
        selection_func=tournament_selection,
        elitism=True,
        n_elite=2
    )

best_candidate = ga.fit(n_iters=20, show_iters=True)
print("Miglior candidato trovato:", best_candidate.values)


# Esecuzione
#ea = EvolutionaryAlgorithm(POP_SIZE, OPTIMIZATION_TRESHOLD, MUTATION_RATE)
#ea.execute(items, NUM_GENERATIONS)


#ea_execution_time = timeit.timeit(stmt="ea.execute(items, NUM_GENERATIONS)",  globals=globals(),  # Accesso alle variabili globali number=10  # Esegue 10 volte per ottenere una media)

#print(f"Tempo medio di esecuzione evolutionary algorithm: {ea_execution_time / 10:.6f} secondi")


#candidate = BitVectorCandidate.generate(POP_SIZE)
#print("Candidate values:", candidate.candidate.tolist())

