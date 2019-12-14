from numpy import diagflat


class GradientDescentOptimizer:
    def __init__(self, activation, learning_rate):
        self.learning_rate = learning_rate
        self.activation = activation

    def activation_derivitive(self, x):
        return diagflat(self.activation(x, derivitive=True))

    def run(self, layer, delta, z, a, weights, biases):

        b_delta = self.learning_rate * self.activation_derivitive(z[layer]).dot(delta)

        biases[layer - 1] -= b_delta
        weights[layer - 1] -= b_delta.dot(a[layer - 1].T)

        delta = None

        if layer != 1:
            delta = weights[layer - 1].T.dot(b_delta)

        return delta


class AdamOptimzer:
    def run():
        return False


"""
from math import sin, cos, pi, sqrt
import random


def func(x, derivitive=False):
    if derivitive:
        return (sin(x) - 0.25) * cos(x)
    return (sin(x) - 0.25) ** 2


def exaple_adam():
    gamma_1, gamma_2, alpha = 0.002, 0.005, 0.0001

    theta = 0.25

    r, p = 0, 0

    print(func(theta))

    for t in range(1, 5000):
        d = func(theta, True)
        g_1 = 1 - (1 - gamma_1) ** t
        g_2 = 1 - (1 - gamma_2) ** t

        r = (1 - gamma_1) * d + gamma_1 * r
        p = (1 - gamma_1) * d ** 2 + gamma_1 * p

        r_hat = r / g_1
        p_hat = p / g_2

        v = alpha * r_hat / sqrt(p_hat)

        theta -= v

    print(func(theta))
    print(theta * 180 / pi)
    print(sin(theta))
"""
