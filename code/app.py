import numpy as np
from base import ROOT_DIR

from mathai.activations import ArcTan, TanH
from mathai.models import NeuralBlock


T = NeuralBlock(ArcTan(), (4, 4), (4, 1))


print(T.forward(np.array([[2], [3], [-1], [4]])))
