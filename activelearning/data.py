import numpy as np
import pandas as pd


class SyntheticData:
    def __init__(self):
        pass

    @staticmethod
    def sample_2_classes(N, centroids, p_z):
        y = np.random.binomial(1, p_z, N)

        X = np.zeros((N, 2))

        X[y == 0, :] = np.random.normal(loc=centroids[0, :], scale=np.array([0.5, 0.5]), size=(len(y[y == 0]), 2))
        X[y == 1, :] = np.random.normal(loc=centroids[1, :], scale=np.array([0.5, 0.5]), size=(len(y[y == 1]), 2))

        df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y})

        return df
