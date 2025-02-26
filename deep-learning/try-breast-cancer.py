from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from TryNeuralNetwork import TryNeuralNetwork
import numpy as np

# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets 


X.iloc[:, 5] = X.iloc[:, 5].replace([None, np.nan], 0)

X = X.to_numpy()
y = y.to_numpy()

y = y.flatten()


# piante, case, macchine
#X = [[20, 5, 3], [0, 10, 30], [25, 1, 0], [3, 0, 1], [0, 40, 13], [29, 2, 1], [2, 5, 7], [1, 4, 17], [20, 170, 120], 
     #[30, 2, 0], [4, 0, 0], [38, 2, 4], [21, 500, 68], [3, 1, 1], [12, 1, 1], [200, 400, 512], [8, 2, 135], [10, 2, 0],
     #[40, 6, 6], [50, 30, 9], [40, 600, 523], [3, 6, 8], [4, 1, 0], [78, 0, 1], [5, 13, 2], [25, 1, 0], [2, 6, 2], 
     #[6, 0, 0], [13, 1, 1], [2, 15, 17], [5, 0, 1]]

# 0 = natura, 1 = città
#y = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
  
# metadata 
#print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
#print(breast_cancer_wisconsin_original.variables) 

#print(X)
#print(y)

neural_network = TryNeuralNetwork() # type: ignore
neural_network.train(X_train, y_train)
y_pred = neural_network.predict(X_test)

print(y_pred)
print(y_test)

