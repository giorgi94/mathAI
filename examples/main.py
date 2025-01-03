import os
import pickle
import sys

import base
import numpy as np

from ann.activations import sigmoid, atan
from ann.models import ANNetwork
from ann.optimizers import GradientDescentOptimizer

np.set_printoptions(precision=16)


def create_data():
    training_data = [
        (np.array([[i]]), np.array([[i**0.5]])) for i in np.linspace(0, 1, 100)
    ]
    testing_data = [np.array([[i]]) for i in np.linspace(0, 1, 5)]

    return training_data, testing_data


def create_model():

    N = ANNetwork(
        activation=sigmoid,
        optimizer=GradientDescentOptimizer(learning_rate=0.001, activation=sigmoid),
    )

    layers = [1, 8, 10, 8, 1]

    N.load_layers(layers)

    return N


def train_model():

    training_data, testing_data = create_data()

    N = create_model()

    # N.load_random_weights()

    N.load()

    N.training(training_data[:], max_steps=10_000, each=10)

    N.dump()


def test_model():
    training_data, testing_data = create_data()
    N = create_model()
    N.load()

    x = 0.38597

    N.forward(np.array([[x]]))

    y = N.output().flatten()[0]

    print(x**0.5)
    print(y)


if __name__ == "__main__":

    train_model()

    # test_model()
