import numpy as np
from .activations import ActivationFunc


class NeuralBlock:

    def __init__(
        self,
        activation: ActivationFunc,
        input_size: int,
        output_size: int,
        weight: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        chain: bool = False,
    ):
        wsize = (output_size, input_size)
        bsize = (output_size, 1)

        self.chain = chain
        self.activation = activation

        if weight is None:
            self.weight = np.random.uniform(-1, 1, size=wsize)
        else:
            assert weight.shape == wsize, f"Weight shape should be {wsize}"
            self.weight = weight

        if bias is None:
            self.bias = np.random.uniform(-1, 1, size=bsize)
        else:
            assert bias.shape == bsize, f"Bias shape should be {bsize}"
            self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:

        return self.activation.calc(self.weight @ x + self.bias)

    def dforward(self, x: np.ndarray) -> np.ndarray:

        return self.activation.dcalc(self.weight @ x + self.bias)

    def backward(self, x: np.ndarray, y: np.ndarray, gamma: float = 0.01):
        y_hat = self.forward(x)
        dy_hat = self.dforward(x)

        D = dy_hat * (y_hat - y)

        self.bias -= gamma * D
        self.weight -= gamma * (D @ x.T)

        if self.chain:
            x -= self.weight.T @ D


class NeuralChain:
    pass
