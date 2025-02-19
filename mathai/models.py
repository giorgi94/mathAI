import numpy as np
from .activations import ActivationFunc


class NeuralBlock:

    weight: np.array
    bias: np.array

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

    def forward(self, x: np.ndarray):

        return self.activation.calc(self.weight @ x + self.bias)
