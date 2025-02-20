import random
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
import timeit

# Parametri
NUM_ITEMS = 10
POP_SIZE = 45
MUTATION_RATE = 0.05
NUM_GENERATIONS = 20
OPTIMIZATION_TRESHOLD = 50


items = [{ "weight": random.randint(1, 50), "value": random.randint(1, 100)} for _ in range(NUM_ITEMS)]
print(items)

# Esecuzione
ea = EvolutionaryAlgorithm(POP_SIZE, OPTIMIZATION_TRESHOLD, MUTATION_RATE)
ea.execute(items, NUM_GENERATIONS)


ea_execution_time = timeit.timeit(
    stmt="ea.execute(items, NUM_GENERATIONS)",  
    globals=globals(),  # Accesso alle variabili globali
    number=10  # Esegue 10 volte per ottenere una media
)

print(f"Tempo medio di esecuzione evolutionary algorithm: {ea_execution_time / 10:.6f} secondi")





def knapsack_bottom_up(weights, values, max_weight):
    n = len(weights)
    dp = [[0] * (max_weight + 1) for _ in range(n + 1)]

    # Costruzione della tabella dp
    for i in range(1, n + 1):
        for w in range(max_weight + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    # Ricostruzione degli oggetti scelti
    w = max_weight
    chosen_items = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:  # Se il valore è cambiato, significa che l'oggetto è stato preso
            chosen_items.append(i - 1)
            w -= weights[i - 1]

    #return dp[n][max_weight], chosen_items[::-1]  # Miglior valore ottenuto e oggetti scelti
    print(f"Valore massimo ottenuto = {dp[n][max_weight]} Soluzione = {chosen_items[::-1]}")



# Dati del problema
weights = [item["weight"] for item in items]
values = [item["value"] for item in items]
max_weight = OPTIMIZATION_TRESHOLD  # Peso massimo dello zaino

knapsack_bottom_up(weights, values, max_weight)

#print(f"Valore massimo ottenuto: {max_value}")
#print(f"Oggetti selezionati (indice): {selected_items}")

bu_execution_time = timeit.timeit(
    stmt="knapsack_bottom_up(weights, values, max_weight)",  
    globals=globals(),  # Accesso alle variabili globali
    number=10  # Esegue 10 volte per ottenere una media
)

print(f"Tempo medio di esecuzione: {bu_execution_time / 10:.6f} secondi")