import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max_val = np.max(z)

        res = []
        total_sum = 0

        # Compute denominator
        for i in z:
            total_sum += math.exp(i - max_val)

        # Compute probabilities
        for i in z:
            probability = math.exp(i - max_val) / total_sum
            res.append(probability)

        return np.round(np.array(res), 4)