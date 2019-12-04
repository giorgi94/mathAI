class GradientDescentOptimizer:
    def run():
        return False


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
