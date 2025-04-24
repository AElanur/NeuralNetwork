import numpy as np

X_TRAIN = np.array([
    [0.8, 0.9, 0.7],
    [0.9, 0.1, 0.0],
    [0.7, 0.2, 0.6],
    [0.6, 0.85, 0.65]
])

Y_TRAIN = np.array([1, 0, 0, 1])

FEATURE_NAMES = ["Sweetness", "Crunchiness", "Redness"]
FRUIT_NAMES = {
    0: "Banana/Orange",
    1: "Apple"
}