import numpy as np
from abc import ABC, abstractmethod


class ActivationFunc(ABC):

    @abstractmethod
    def calc(self, x): ...

    @abstractmethod
    def dcalc(self, x): ...


class TanH(ActivationFunc):

    def calc(self, x):
        return np.tanh(x)

    def dcalc(self, x):
        return 1 / np.cosh(x) ** 2


class ArcTan(ActivationFunc):

    def calc(self, x):
        return np.arctan(x)

    def dcalc(self, x):
        return 1 / (1 + x**2)


class Line(ActivationFunc):

    def calc(self, x):
        return x

    def dcalc(self, x):
        return 1
