from typing import Tuple

import numpy as np


class EnvInterface:
    def reset(self) -> Tuple[np.ndarray, bool]:
        pass

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        pass
