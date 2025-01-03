from numpy import exp, atan as _atan


def sigmoid(x, derivitive=False):
    if derivitive:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + exp(-x))


def atan(x, derivitive=False):
    if derivitive:
        return 1 / (1 + x**2)
    return _atan(x)
