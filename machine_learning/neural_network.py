import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=3, hidden_size=4, output_size=1):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.sigmoid(self.z2)

    def train(self, x, y, epochs=1000, lr=0.01):
        for _ in range(epochs):
            output = self.forward(x)

            error = output - y.reshape(-1, 1)
            dW2 = np.dot(self.a1.T, error)
            db2 = np.sum(error, axis=0)

            error_hidden = np.dot(error, self.w2.T) * (self.z1 > 0)
            dW1 = np.dot(x.T, error_hidden)
            db1 = np.sum(error_hidden, axis=0)

            self.w2 -= lr * dW2
            self.b2 -= lr * db2
            self.w1 -= lr * dW1
            self.b1 -= lr * db1

