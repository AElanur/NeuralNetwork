import numpy as np
from perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

print("\nPredictions:")
for xi in X:
    prediction = ppn.predict(xi)
    print(f"Input {xi} -> Predictions: {prediction}")

print("\nFinal weights:", ppn.w_)
