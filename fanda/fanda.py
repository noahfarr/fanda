from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Fanda:
    fig: Optional[plt.Figure] = None
    ax: Optional[plt.Axes] = None
    axs: Optional[np.ndarray] = field(default=None, repr=False)

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def select(self, index):
        self.ax = self.axs.flat[index]
        return self
