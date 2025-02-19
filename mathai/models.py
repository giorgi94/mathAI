import numpy as np
from .activations import ActivationFunc


class NeuralBlock:

    weight: np.ndarray
    bias: np.ndarray

    activation: ActivationFunc

    def __init__(
        self,
        activation: ActivationFunc,
        wsize: tuple[int, int],
        bsize: tuple[int, int],
        weight: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ):

        self.activation = activation

        if weight is None:
            self.weight = np.random.uniform(-1, 1, size=wsize)
        else:
            self.weight = weight

        if bias is None:
            self.bias = np.random.uniform(-1, 1, size=bsize)
        else:
            self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:

        return self.activation.calc(self.weight @ x + self.bias)

    def dforward(self, x: np.ndarray) -> np.ndarray:

        return self.activation.dcalc(self.weight @ x + self.bias)

    def backward(self, x: np.ndarray, y: np.ndarray, gamma: float = 0.01):
        y_hat = self.forward(x)

        dy_hat = np.diagflat(self.dforward(x))

        y_delta = y_hat - y

        self.bias -= gamma * (dy_hat @ y_delta)
        self.weight -= gamma * (dy_hat @ y_delta @ x.T)
