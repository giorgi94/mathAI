import os
import random
import pickle
import numpy as np

from .optimizers import GradientDescentOptimizer


def assure_path_exists(path, isfile=True):
    if isfile:
        dir_path = os.path.dirname(path)
    else:
        dir_path = path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class ANNetworkBase:
    def load_layers(self, layers):
        self.layers = layers
        self.last = len(layers) - 1

    def load_random_weights(self):
        self.biases = []
        self.weights = []

        n0 = self.layers[0]

        for n1 in self.layers[1:]:
            self.weights.append(2 * np.random.random((n1, n0)) - 1)
            self.biases.append(2 * np.random.random((n1, 1)) - 1)
            n0 = n1

    def forward(self, X):
        pass

    def backward(self, Y):
        pass

    def output(self):
        pass

    def check_error_norm(self, Y):
        return np.linalg.norm(Y - self.output())

    def train(self, X, Y):
        self.forward(X)
        self.backward(Y)

    def training(self, table, max_steps, each=1):

        table_data = table[:]

        for step in range(max_steps):
            error = []
            for row in table_data:
                X, Y = row
                self.train(X, Y)
                error.append(self.check_error_norm(Y))

            if step % each == 0:
                print(sum(error) / len(error))

            random.shuffle(table_data)

    def dump(self, path="dump.pkl"):
        path = os.path.abspath(path)
        assure_path_exists(path)

        with open(path, "wb") as f:
            pickle.dump((self.weights, self.biases), f)

    def load(self, path="dump.pkl"):
        path = os.path.abspath(path)

        with open(path, "rb") as f:
            self.weights, self.biases = pickle.load(f)


class ANNetwork(ANNetworkBase):
    def __init__(self, activation, optimizer):
        self.activation = activation
        self.optimizer = optimizer

    def forward(self, X):
        self.z = [None]
        self.a = [X]

        layer = 0
        for w, b in zip(self.weights, self.biases):
            layer += 1
            self.z.append(w.dot(self.a[layer - 1]) + b)
            self.a.append(self.activation(self.z[layer]))

    def backward(self, Y):
        delta = self.output() - Y

        for layer in range(self.last, 0, -1):

            delta = self.optimizer.run(
                layer, delta, self.z, self.a, self.weights, self.biases
            )

    def output(self):
        return self.a[-1]
