import neat as nt
import numpy as np

# Valore ideale che vogliamo ottenere con la nostra rete neurale
IDEAL_FITNESS = 1.0

# Funzione di valutazione della rete neurale
def evaluate_genome(genome, config):
    net = nt.nn.FeedForwardNetwork.create(genome, config)
    
    # Testiamo la rete con alcuni input di esempio
    inputs = np.linspace(-1, 1, 10)  # 10 valori tra -1 e 1
    expected_outputs = np.sin(inputs)  # Ad esempio, vogliamo approssimare una sinusoide
    
    fitness = 0.0
    for x, target in zip(inputs, expected_outputs):
        output = net.activate([x])[0]  # La rete restituisce un valore
        fitness += 1.0 - abs(target - output)  # Penalizza deviazioni

    # Normalizziamo la fitness per ottenere un valore tra 0 e 1
    fitness /= len(inputs)
    
    return fitness

# Funzione di valutazione per l'intera popolazione
def evaluate_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)

# Configurazione di NEAT
def run_neat():
    config_path = "deep-learning/config-neat.txt"
    config = nt.Config(nt.DefaultGenome, nt.DefaultReproduction,
                         nt.DefaultSpeciesSet, nt.DefaultStagnation,
                         config_path)

    # Creazione della popolazione
    population = nt.Population(config)
    
    # Output statistico
    population.add_reporter(nt.StdOutReporter(True))
    stats = nt.StatisticsReporter()
    population.add_reporter(stats)

    # Evoluzione della rete neurale
    winner = population.run(evaluate_population, 100)  # 100 generazioni max

    print("\nRete neurale ottimale trovata!")
    print(f"Fitness ottimale: {winner.fitness}")

    return winner

# Avvia l'evoluzione
if __name__ == "__main__":
    best_network = run_neat()
