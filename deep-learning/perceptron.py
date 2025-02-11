import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(input_size + 1)  # +1 per il bias

    def activation(self, x):
        return self.relu_activation(x)
    
    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def relu_activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        X = np.insert(X, 0, 1)  # Aggiunge il bias (X0 = 1)
        return self.activation(np.dot(self.weights, X))
    
    def get_predictions(self, X):
        y = np.array([])
        for x in X:
            pred = self.predict(x)
            y = np.append(y, pred)
        return np.array(y, dtype="int")
    
    def accuracy(self, y_pred, y_test):
        pred_len = len(y_pred)
        test_len = len(y_test)
        if(pred_len != test_len): return 0
        else: return np.sum(y_pred == y_test) / pred_len

    def train(self, X_train, y_train):
        for _ in range(self.epochs):
            for X, y in zip(X_train, y_train):
                X = np.insert(X, 0, 1)  # Aggiunge il bias (X0 = 1)
                y_pred = self.activation(np.dot(self.weights, X))
                self.weights += self.learning_rate * (y - y_pred) * X  # Aggiornamento dei pesi

# Esempio di utilizzo
if __name__ == "__main__":
    #feature 1: 1 = lunedi, 2 = martedi, 3 = mercoledi ...
    #feature 2: 1 = sole, 2 = nuvoloso, 3 = pioggia, 4 = neve
    #target: 0 = poche persone, 1 = tante persone

    # Dati di training: AND logic gate
    X_train = np.array([[1, 1], [2, 2],  [1, 3], [6, 3], [7, 2], [4, 1], [7, 1], [5, 1], [2, 4], [7, 3], [4, 1], [3, 2], [3, 3], [1, 1], [2, 1], [5, 2], [5, 3], [2, 4], [5, 4], [1, 4]])
    y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])  # Output per AND

    X_test = np.array([[4, 4], [2, 1], [1, 4], [7, 4], [1, 4], [7, 3], [4, 2], [6, 1], [3, 2], [4, 1], [5, 2], [3, 4], [2, 1]])
    y_test = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1])
    # Inizializzazione del percettrone
    perc = Perceptron(input_size=2, learning_rate=0.1, epochs=20)
    
    # Addestramento
    perc.train(X_train, y_train)

    y_pred = perc.get_predictions(X_test)
    print(y_pred)
    print(y_test)
    print(perc.accuracy(y_pred, y_test))
