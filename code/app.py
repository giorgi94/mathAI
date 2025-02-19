import numpy as np
from base import ROOT_DIR

from mathai.activations import ArcTan, TanH
from mathai.models import NeuralBlock


def example_1():
    weight = np.random.rand(5, 4)
    bias = np.random.rand(5, 1)

    T0 = NeuralBlock(ArcTan(), (5, 4), (5, 1), weight=weight, bias=bias)

    X_list = [np.random.rand(4, 1) for _ in range(10)]
    Y_list = [T0.forward(x) for x in X_list]

    T = NeuralBlock(ArcTan(), (5, 4), (5, 1))

    for _ in range(50_000):
        for x, y in zip(X_list, Y_list):
            T.backward(x, y)

    for x, y in zip(X_list, Y_list):
        print(((T.forward(x) - y) ** 2).sum() ** 0.5 / 2)


def example_2():
    weight_1 = np.random.rand(5, 4)
    bias_1 = np.random.rand(5, 1)

    weight_2 = np.random.rand(3, 5)
    bias_2 = np.random.rand(3, 1)

    T1 = NeuralBlock(ArcTan(), (5, 4), (5, 1), weight=weight_1, bias=bias_1)
    T2 = NeuralBlock(ArcTan(), (3, 5), (3, 1), weight=weight_2, bias=bias_2, chain=True)

    T = lambda x: T2.forward(T1.forward(x))

    X_list = [np.random.rand(4, 1) for _ in range(10)]
    Y_list = [T(x) for x in X_list]

    M1 = NeuralBlock(ArcTan(), (5, 4), (5, 1))
    M2 = NeuralBlock(ArcTan(), (3, 5), (3, 1), chain=True)

    M = lambda x: M2.forward(M1.forward(x))

    for _ in range(60_000):
        for x, y in zip(X_list, Y_list):
            t = M1.forward(x)
            M2.backward(t, y)
            M1.backward(x, t)

    for x, y in zip(X_list, Y_list):
        print(((M(x) - y) ** 2).sum() ** 0.5 / 2)


if __name__ == "__main__":
    # example_1()
    example_2()
