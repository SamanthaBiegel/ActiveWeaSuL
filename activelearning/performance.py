import numpy as np
import torch


class ModelPerformance:
    def __init__(self, df):
        self.df = df

    def accuracy(self):
        """Compute overall accuracy from label predictions"""

        probs = self.predict()
        y = self.df["y"].values

        return self._accuracy(probs, y)

    def _accuracy(self, prob_labels, y):

        return (torch.argmax(prob_labels, dim=1) == torch.Tensor(y)).sum().item() / len(y)