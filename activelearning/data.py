import numpy as np
import pandas as pd


class SyntheticData:
    def __init__(self, N, p_z, centroids):
        self.N = N
        self.p_z = p_z
        self.centroids = centroids

    def sample_y(self):
        
        y = np.random.binomial(1, self.p_z, self.N)

        return y

    def sample_features(self, y):

        X = np.zeros((self.N, 2))

        X[y == 0, :] = np.random.normal(loc=self.centroids[0, :], scale=np.array([0.5, 0.5]), size=(len(y[y == 0]), 2))
        X[y == 1, :] = np.random.normal(loc=self.centroids[1, :], scale=np.array([0.5, 0.5]), size=(len(y[y == 1]), 2))

        return X

    def sample_data_set(self):

        self.y = self.sample_y()
        self.X = self.sample_features(self.y)

        return self

    def apply_lf(self, wl_dict):

        self.wl_df = pd.DataFrame()

        for axis, wls in wl_dict.items():
            for i, wl in enumerate(wls):
                self.wl_df["wl_"+str(axis)+"_"+str(i)] = (self.X[:, axis] < wl)*1

        return self

    def create_df(self):

        df = pd.DataFrame({'x1': self.X[:, 0], 'x2': self.X[:, 1], 'y': self.y})
        # df = pd.concat([df, self.wl_df], axis=1)

        return df
