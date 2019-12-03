import os
import sys
import pickle
import numpy as np

from ann_model import ANNetwork, sigmoid

np.set_printoptions(precision=16)

with open("data/imgvec.pkl", "rb") as f:
    data = pickle.load(f)


training_data = data[:-10]
testing_data = data[-10:]

N = ANNetwork()

N.activation = sigmoid

N.learning_rate = 0.25

layers = [784, 36, 36, 36, 10]


N.load_layers(layers)
N.load_random_weights()

N.load()


# N.training(training_data[:], max_steps=2000, each=100)

# N.dump()


print("\ncheck:")


for X, Y in training_data[:40]:
    N.forward(X)
    print(N.output().argmax(), Y.argmax())
