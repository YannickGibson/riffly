from abc import abstractmethod

import numpy as np
import torch
from torch import nn


class ModelInterface(nn.Module):
    def __init__(self, columns, rows) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows

    def generate(self, seed: int | None = None, threshold: float = None) -> np.ndarray:
        """Generates a new matrix using the model's _generate() method with options."""
        if seed is not None:
            torch.manual_seed(seed)
        if threshold is None:
            threshold = 0.5
        return self._generate() > threshold

    @abstractmethod
    def _generate(self) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, x) -> torch.Tensor:
        pass
