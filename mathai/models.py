import numpy as np
from typing import Callable


class NeuralBlock:

    weight: np.array
    bias: np.array

    activation: Callable

    def __init__(
        self,
        wsize: tuple[int, int],
        bsize: tuple[int, int],
        weight: np.array | None = None,
        bias: np.array | None = None,
    ):
        if weight is None:
            self.weight = np.random.uniform(-1, 1, size=wsize)
        else:
            self.weight = weight

        if bias is None:
            self.bias = np.random.uniform(-1, 1, size=bsize)
        else:
            self.bias = bias
