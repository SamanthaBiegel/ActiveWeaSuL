import numpy as np


class ModelPerformance:
    def __init__(self, df):
        self.df = df

    def accuracy(self):
        """Compute overall accuracy from label predictions"""

        probs = self.predict()
        y = self.df["y"].values

        return self._accuracy(probs, y)

    def _accuracy(self, prob_labels, y):
        prob_labels_numpy = prob_labels.detach().clone().numpy()

        return (np.argmax(np.around(prob_labels_numpy), axis=-1) == np.array(y)).sum() / len(y)