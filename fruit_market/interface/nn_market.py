import numpy as np
from fruit_market.machine_learning.neural_network import NeuralNetwork
from emotion_detection.data import FEATURE_NAMES
from emotion_detection.data import  load_fruit_data

X_TRAIN, Y_TRAIN, cfg = load_fruit_data()

nn = NeuralNetwork(
    input_size=cfg["model"]["parameters"]["input_size"],
    hidden_size=cfg["model"]["parameters"]["hidden_size"],
    output_size=cfg["model"]["parameters"]["output_size"],
)

nn.train(
    X_TRAIN,
    Y_TRAIN,
    epochs=cfg["training"]["epochs"],
    lr=cfg["model"]["parameters"]["learning_rate"]
)

def predict_fruit(features):
    prob = nn.forward(features.reshape(1, -1))[0][0]
    return 1 if prob > 0.5 else 0

def get_fruit_features():
    print("\nEnter fruit characteristics (0-10 scale): ")
    features = []
    for feature_name in FEATURE_NAMES:
        value = float(input(f"{feature_name} (0-10): ")) / 10
        features.append(value)
    return np.array(features)