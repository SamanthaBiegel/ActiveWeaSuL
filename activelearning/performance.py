import numpy as np
import torch


class PerformanceMixin:
    """Mixin that adds analysis functionality to models for computing performance measures"""

    def analyze(self):
        """Get predictions and analyze performance"""

        self.y = self.df["y"].values

        self._analyze(self.preds, self.y)

    def _analyze(self, prob_labels, y):

        predicted_labels = torch.argmax(prob_labels, dim=1).cpu().detach().numpy()

        y_set = list(range(prob_labels.shape[1]))
        self.TN, self.FN, self.FP, self.TP = (((predicted_labels == i) & (y == j)).sum() for i in y_set for j in y_set)

        self.metric_dict = {"Labels:": self.model_name, "MCC": self.MCC(), "Precision": self.precision(), "Recall": self.recall(), "Accuracy": self.accuracy()}

    def accuracy(self):
        """Compute overall accuracy from label predictions"""

        return self._accuracy(self.preds, self.y)

    def _accuracy(self, prob_labels, y):

        return (torch.argmax(prob_labels, dim=1).cpu().detach().numpy() == y).sum() / len(y)

    def MCC(self):
        """Matthews correlation coefficient"""

        nominator = self.TP * self.TN - self.FP * self.FN
        denominator_squared = (self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)

        return nominator / np.sqrt(denominator_squared)

    def recall(self):
        """Fraction of true class 1 and all class 1"""

        return self.TP / (self.TP + self.FN)

    def precision(self):
        """Fraction of true class 1 and predicted class 1"""

        return self.TP / (self.TP + self.FP)

    def print_metrics(self):
        """Pretty print metric dict"""

        for key, value in self.metric_dict.items():
            print(key, ': ', value)