import numpy as np
import torch


class PerformanceMixin:
    """Mixin that adds analysis functionality to models for computing performance measures"""

    def analyze(self):
        """Get predictions and analyze performance"""

        self.y = self.df["y"].values

        self.metric_dict = self._analyze(self.preds, self.y)
        self.metric_dict["Labels"] = self.model_name

    def _analyze(self, prob_labels, y):

        predicted_labels = torch.argmax(prob_labels, dim=1).cpu().detach().numpy()

        y_set = list(range(prob_labels.shape[1]))
        TN, FN, FP, TP = (((predicted_labels == i) & (y == j)).sum() for i in y_set for j in y_set)

        return {"MCC": self.MCC(TP, TN, FP, FN),
                "Precision": self.precision(TP, FP),
                "Recall": self.recall(TP, FN),
                "Accuracy": self._accuracy(prob_labels, y)}

    def accuracy(self):
        """Compute overall accuracy from label predictions"""

        return self._accuracy(self.preds, self.y)

    def _accuracy(self, prob_labels, y):

        return (torch.argmax(prob_labels, dim=1).cpu().detach().numpy() == y).sum() / len(y)

    def MCC(self, TP, TN, FP, FN):
        """Matthews correlation coefficient"""

        nominator = TP * TN - FP * FN
        denominator_squared = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)

        return nominator / np.sqrt(denominator_squared)

    def recall(self, TP, FN):
        """Fraction of true class 1 and all class 1"""

        return TP / (TP + FN)

    def precision(self, TP, FP):
        """Fraction of true class 1 and predicted class 1"""

        return TP / (TP + FP)

    def print_metrics(self):
        """Pretty print metric dict"""

        for key, value in self.metric_dict.items():
            print(key, ': ', value)