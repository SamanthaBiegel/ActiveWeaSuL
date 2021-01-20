import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


class SyntheticDataset(TensorDataset):
    """Synthetic Dataset"""

    def __init__(self, df: pd.DataFrame, Y: torch.Tensor) -> None:
        self.X = df.loc[:, ["x1", "x2"]]
        self.Y = Y

    def __getitem__(self, index: int):
        return torch.Tensor(self.X.iloc[index]), self.Y[index]

    def __len__(self):
        return len(self.X)


class SyntheticDataGenerator:
    """Generate synthetic dataset"""
    
    def __init__(self, N, p_z, centroids, seed=456):
        self.N = N
        self.p_z = p_z
        self.centroids = centroids
        np.random.seed(seed)

    def sample_y(self):
        y = np.random.binomial(1, self.p_z, self.N)
        return y

    def sample_features(self, y):
        X = np.zeros((self.N, 2))

        X[y == 0, :] = np.random.normal(loc=self.centroids[0, :], scale=np.array([0.5, 0.5]), size=(len(y[y == 0]), 2))
        X[y == 1, :] = np.random.normal(loc=self.centroids[1, :], scale=np.array([0.5, 0.5]), size=(len(y[y == 1]), 2))

        return X

    def sample_dataset(self):
        self.y = self.sample_y()
        self.X = self.sample_features(self.y)

        return self

    def create_df(self):
        df = pd.DataFrame({'x1': self.X[:, 0], 'x2': self.X[:, 1], 'y': self.y})

        return df