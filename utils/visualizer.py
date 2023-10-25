from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self):
        pass

    def draw_flow(self, flow: np.ndarray) -> Visualizer:
        fig, ax = plt.subplots()
        ax.imshow(flow)
        plt.show()

        return self
