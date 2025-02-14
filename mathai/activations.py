import numpy as np


class TanH:

    def calc(self, x):
        return np.tanh(x)

    def dcalc(self, x):
        return 1 / np.cosh(x) ** 2


class ArcTan:

    def calc(self, x):
        return np.atan(x)

    def dcalc(self, x):
        return 1 / (1 + x**2)


# def line(x, diff=False):
#     if diff:
#         return 1
#     return x


# def leaky_relu(x, alpha=0.01, diff=False):

#     if x >= 0:
#         return x if diff else 1

#     return alpha * x if diff else alpha


# def elu(x, alpha=0.01, diff=False):

#     if x >= 0:
#         return x if diff else 1

#     return alpha * (np.exp(x) - 1) if diff else alpha * np.exp(x)


# def sigmoid(x, derivitive=False):
#     if derivitive:
#         return sigmoid(x) * (1 - sigmoid(x))
#     return 1 / (1 + np.exp(-x))


# def atan(x, derivitive=False):
#     if derivitive:
#         return 1 / (1 + x**2)
#     return np.atan(x)
