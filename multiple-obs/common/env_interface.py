import numpy as np
from typing import Tuple


class EnvInterface:
    def reset(self) -> Tuple[np.ndarray, bool]:
        pass

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        pass
