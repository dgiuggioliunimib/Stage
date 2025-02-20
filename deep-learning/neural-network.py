import random
import math

# **1️⃣ Classe PerceptronLayer (uno strato di percettroni)**
class PerceptronLayer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_neurons)]
        self.biases = [random.uniform(-1, 1) for _ in range(num_neurons)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = []
        for i in range(self.num_neurons):
            activation = sum(self.inputs[j] * self.weights[i][j] for j in range(self.num_inputs)) + self.biases[i]
            self.outputs.append(self.sigmoid(activation))
        return self.outputs

    def backward(self, error, learning_rate):
        self.deltas = [error[i] * self.sigmoid_derivative(self.outputs[i]) for i in range(self.num_neurons)]
        for i in range(self.num_neurons):
            for j in range(self.num_inputs):
                self.weights[i][j] += learning_rate * self.deltas[i] * self.inputs[j]
            self.biases[i] += learning_rate * self.deltas[i]
        return [sum(self.weights[i][j] * self.deltas[i] for i in range(self.num_neurons)) for j in range(self.num_inputs)]

# **2️⃣ Classe NeuralNetwork (rete completa)**
class NeuralNetwork:
    def __init__(self):
        self.hidden_layer = PerceptronLayer(2, 2)  # Strato nascosto con 2 neuroni
        self.output_layer = PerceptronLayer(2, 1)  # Strato di output con 1 neurone
    
    def forward(self, inputs):
        hidden_outputs = self.hidden_layer.forward(inputs)
        final_output = self.output_layer.forward(hidden_outputs)
        return final_output

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # **Fase Forward**
                hidden_outputs = self.hidden_layer.forward(X[i])
                final_output = self.output_layer.forward(hidden_outputs)

                # **Errore**
                output_error = [y[i] - final_output[0]]
                total_error += abs(output_error[0])

                # **Fase Backpropagation**
                hidden_error = self.output_layer.backward(output_error, learning_rate)
                self.hidden_layer.backward(hidden_error, learning_rate)

            # Stampa dell'errore ogni 1000 epoche
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Errore: {total_error:.4f}")

    def predict(self, inputs):
        return self.forward(inputs)[0]

# **3️⃣ Dati per XOR**
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]  # Output XOR

# **4️⃣ Creiamo e addestriamo la rete**
nn = NeuralNetwork()
nn.train(X, y)

# **5️⃣ Testiamo la rete**
print("\nTest della rete:")
for i in range(len(X)):
    output = nn.predict(X[i])
    print(f"Input: {X[i]}, Output predetto: {output:.4f}")
