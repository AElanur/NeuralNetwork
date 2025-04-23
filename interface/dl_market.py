import numpy as np
from machine_learning.neural_network import NeuralNetwork

nn = NeuralNetwork(input_size=3, hidden_size=4)
x_train = np.array([
    [0.8, 0.9, 0.7], # Apple
    [0.9, 0.1, 0.0], # Banana
    [0.7, 0.2, 0.6],  # Orange
    [0.6, 0.85, 0.65] # Sour apple
])

y_train = np.array([1, 0, 0, 1])
nn.train(x_train, y_train, epochs=1000)

def predict_fruit(features):
    prob = nn.forward(features.reshape(1, -1))[0][0]
    return "Like" if prob > 0.5 else "Dislike"

fruit_names = {
    0: "Banana/Orange",
    1: "Apple"
}

def get_fruit_features():
    print("\nEnter fruit characteristics (0-10 scale): ")
    sweetness = float(input("Sweetness (0-10): ")) / 10
    crunchiness = float(input("Crunchiness (0-10): ")) / 10
    redness = float(input("Redness (0-10): ")) / 10
    return np.array([sweetness, crunchiness, redness])