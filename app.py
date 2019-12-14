import os
import sys
import pickle
import numpy as np

from ann.models import ANNetwork
from ann.activations import sigmoid
from ann.optimizers import GradientDescentOptimizer

np.set_printoptions(precision=16)

with open("data/imgvec.pkl", "rb") as f:
    data = pickle.load(f)


training_data = data[:-10]
testing_data = data[-10:]

N = ANNetwork(
    activation=sigmoid,
    optimizer=GradientDescentOptimizer(learning_rate=0.5, activation=sigmoid),
)


layers = [784, 36, 36, 36, 10]


N.load_layers(layers)
N.load_random_weights()

N.load()


N.training(training_data[:], max_steps=100, each=10)

N.dump()


# print("\ncheck:")


# for X, Y in training_data[:40]:
#     N.forward(X)
#     print(N.output().argmax(), Y.argmax())
