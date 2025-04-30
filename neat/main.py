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

'''
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
X_train = X
y_train = y_true_values
X_test = X
y_test = y_true_values

'''
y_true_values = TargetTransform.get_true_values(y)
test_size = 0.3
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

SETTING = NeatSetting(population=150, generations=1000, error=0.02, fitness=LogarithmSquaredError())


optimizer = NeatOptimizer(SETTING, OUTPUT_SIZE, INPUT_SIZE)
network = optimizer.optimize(X_train, y_train)

output = network.forward(X_test)
for i in range(len(X_test)):
    fitness = SETTING.FITNESS.calculate(network, [X_test[i]], [y_test[i]])
    print(f"FITNESS Best Network on test input[{i}]: {fitness}, output: {output[i][0]}, target: {y_test[i]}")
# X, y_true_values



# Note:
# La determinazione di quanti input appartengono alla network adesso si fa alla init della classe Network
# ma può essere fatta semplicemente anche alla chiamata della forward (poche righe di codice)

# La classe graph permette di ottenere in input alla creazione una lista di foglie. questo è utile nei casi in cui è
# necessario avere già il numero di foglie corretto per la nostra rete neurale (no random)

# La complessità della forward è elevata perchè la rete non è completa tra ogni layer, quindi dinamicamente
# occorre inoltrare nodo per nodo i valori, rispettando anche il rank che si è determinato

# La forward accetta come input una matrice