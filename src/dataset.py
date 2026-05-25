import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ICUTabularDataset(Dataset):
    """Tabular dataset for ICU mortality prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
