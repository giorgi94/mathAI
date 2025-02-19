import numpy as np
from base import ROOT_DIR

from mathai.activations import ArcTan, TanH
from mathai.models import NeuralBlock


weight = np.random.rand(4, 4)
bias = np.random.rand(4, 1)

T0 = NeuralBlock(ArcTan(), (4, 4), (4, 1), weight=weight, bias=bias)

X_list = [np.random.rand(4, 1) for _ in range(10)]
Y_list = [T0.forward(x) for x in X_list]

T = NeuralBlock(ArcTan(), (4, 4), (4, 1))


for _ in range(50_000):
    for x, y in zip(X_list, Y_list):
        T.backward(x, y)

for x, y in zip(X_list, Y_list):
    print(((T.forward(x) - y) ** 2).sum() ** 0.5 / 2)
