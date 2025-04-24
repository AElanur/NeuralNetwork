import numpy as np
from fruit_market.machine_learning.perceptron import Perceptron

x_train = np.array([
    [0.8, 0.9, 0.7], # Apple
    [0.9, 0.1, 0.0], # Banana
    [0.7, 0.2, 0.6],  # Orange
    [0.6, 0.85, 0.65] # Sour apple
])

y_train = np.array([1, -1, -1, 1])

ppn = Perceptron(eta=0.1, n_iter=100)
ppn.fit(x_train, y_train)

fruit_names = {
    -1: "Banana/Orange",
    1: "Apple"
}

def get_fruit_features():
    print("\nEnter fruit characteristics (0-10 scale): ")
    sweetness = float(input("Sweetness (0-10): ")) / 10
    crunchiness = float(input("Crunchiness (0-10): ")) / 10
    redness = float(input("Redness (0-10): ")) / 10
    return np.array([sweetness, crunchiness, redness])