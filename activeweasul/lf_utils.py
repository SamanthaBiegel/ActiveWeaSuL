import numpy as np
import pandas as pd


def apply_lfs(df, lfs):
    L = []

    for labeling_function in lfs:
        L.append(df.apply(labeling_function, axis=1))

    return pd.concat(L, axis=1).values


def analyze_lfs(L, y, lfs):
    lf_stats = pd.DataFrame((pd.DataFrame(L).apply(lambda x: np.where(x == y, True, False), axis=0)).T.mean(axis=1), columns=["Accuracy"])
    lf_stats.index = [lf.__name__ for lf in lfs]
    lf_stats["Coverage"] = (L != -1).mean(axis=0)

    return lf_stats