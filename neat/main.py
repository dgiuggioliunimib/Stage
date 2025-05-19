from Node import Node
from Connection import Connection, InnovationTracker
from Graph import Graph
from Network import Network
from typing import List, Optional
import random as rd
import numpy as np
from Fitness import Fitness, AverageError, LogarithmSquaredError
from Activation import Activation, Sigmoid, SoftMax, ReLU, Identity
import matplotlib.pyplot as plt
from NeatOptimizer import NeatOptimizer, TargetTransform
from NeatSetting import NeatSetting

#'''
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
#'''



EXECUTIONS = 1

generations_history = []
intermediate_nodes_history = []
connections_history = []
min_intermediate_nodes = 0
best_network = None
best_fitness = 0
for _ in range(EXECUTIONS):

    '''
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    y_true_values = TargetTransform.get_true_values(y)
    X_train = X
    y_train = y_true_values
    X_test = X
    y_test = y_true_values

    '''
    y_true_values = TargetTransform.get_true_values(y)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y_true_values, test_size= test_size, random_state=100)
    print("train set: ", X_train)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #'''

    OUTPUT_SIZE = len(set(y))
    INPUT_SIZE = len(X[0])
    #FITNESS = LogarithmSquaredError()
    FITNESS = AverageError()
    SETTING = NeatSetting(population=150, generations=1000, error=0.02, fitness=FITNESS)    

    optimizer = NeatOptimizer(SETTING, OUTPUT_SIZE, INPUT_SIZE)
    network, generations = optimizer.optimize(X_train, y_train)
    generations_history.append(generations)
    intermediate_nodes = len(network.graph.get_intermediate_nodes())
    intermediate_nodes_history.append(intermediate_nodes)
    connections_history.append(len([x for x in network.graph.connections if x.enabled]))
    if min_intermediate_nodes == 0 or intermediate_nodes < min_intermediate_nodes:
        min_intermediate_nodes = intermediate_nodes
        best_network = network
        best_fitness = optimizer.setting.FITNESS.calculate(network, X_test, y_test)
    elif intermediate_nodes == min_intermediate_nodes:
        new_fitness = optimizer.setting.FITNESS.calculate(network, X_test, y_test)
        if new_fitness > best_fitness:
            best_network = network
            best_fitness = new_fitness
    
    output = network.forward(X_test)

    correct_classifications = 0
    for i in range(len(X_test)):
        fitness = SETTING.FITNESS.calculate(network, [X_test[i]], [y_test[i]])
        if fitness > 0.5:
            correct_classifications += 1
        print(f"FITNESS Best Network on test input[{i}]: {fitness}, output: {output[i][0]}, target: {y_test[i]}")
    print(f"Correct classifications: {correct_classifications} / {len(X_test)} ({correct_classifications/len(X_test) * 100} %)")

print("Best network: \nFitness: ", best_fitness, "\nAdjacency list: ", best_network.graph.adjacency_list, "\nConnections: ", best_network.graph.connections)

plt.plot(generations_history)
plt.xlabel("Execution")
plt.ylabel("Generations for result for execution")
plt.title("Gnerations for result")
plt.show()

plt.plot(intermediate_nodes_history)
plt.xlabel("Execution")
plt.ylabel("Intermediate nodes")
plt.title("Intermediate nodes for execution")
plt.show()

plt.plot(connections_history)
plt.xlabel("Execution")
plt.ylabel("Connections")
plt.title("Connections for execution")
plt.show()

sum_gen = 0
sum_int = 0
sum_con = 0
for i in range(EXECUTIONS):
    sum_gen += generations_history[i]
    sum_int += intermediate_nodes_history[i]
    sum_con += connections_history[i]

avg_generations = sum_gen / EXECUTIONS
avg_intermediate_nodes = sum_int / EXECUTIONS
avg_connections = sum_con / EXECUTIONS

print(f"Average Generations: {avg_generations} \nAverage Intermediate Nodes: {avg_intermediate_nodes} \nAverage Connections: {avg_connections}")


'''
output = network.forward(X_test)

for i in range(len(X_test)):
    fitness = SETTING.FITNESS.calculate(network, [X_test[i]], [y_test[i]])
    print(f"FITNESS Best Network on test input[{i}]: {fitness}, output: {output[i][0]}, target: {y_test[i]}")

'''
# X, y_true_values



# Note:
# La determinazione di quanti input appartengono alla network adesso si fa alla init della classe Network
# ma può essere fatta semplicemente anche alla chiamata della forward (poche righe di codice)

# La classe graph permette di ottenere in input alla creazione una lista di foglie. questo è utile nei casi in cui è
# necessario avere già il numero di foglie corretto per la nostra rete neurale (no random)

# La complessità della forward è elevata perchè la rete non è completa tra ogni layer, quindi dinamicamente
# occorre inoltrare nodo per nodo i valori, rispettando anche il rank che si è determinato

# La forward accetta come input una matrice