from numpy import exp


def sigmoid(x, derivitive=False):
    if derivitive:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + exp(-x))
