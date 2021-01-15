import numpy as np
import torch


class PerformanceMixin:
    """Mixin that adds analysis functionality to models for computing performance measures"""

    def analyze(self):
        """Get predictions and analyze performance"""

        self.y = self.y_true

        self.metric_dict = self._analyze(self.prob_labels_train, self.y)
        self.metric_dict["Labels"] = self.model_name

    def _analyze(self, prob_labels, y):

        predicted_labels = torch.argmax(prob_labels, dim=1).cpu().detach().numpy()

        y_set = list(range(prob_labels.shape[1]))
        TN, FN, FP, TP = (((predicted_labels == i) & (y == j)).sum() for i in y_set for j in y_set)

        return {"MCC": self.MCC(TP, TN, FP, FN),
                "Precision": self.precision(TP, FP),
                "Recall": self.recall(TP, FN),
                "Accuracy": self._accuracy(prob_labels, y),
                "F1": self.F1(TP, FP, FN)}

    def accuracy(self):
        """Compute overall accuracy from label predictions"""

        return self._accuracy(self.prob_labels_train, self.y)

    def _accuracy(self, prob_labels, y):

        return (torch.argmax(prob_labels, dim=1).cpu().detach().numpy() == y).sum() / len(y)

    def MCC(self, TP, TN, FP, FN):
        """Matthews correlation coefficient"""

        nominator = TP * TN - FP * FN
        denominator = np.sqrt(TP + FP) * np.sqrt(TP + FN) * np.sqrt(TN + FP) * np.sqrt(TN + FN)

        return nominator / denominator

    def recall(self, TP, FN):
        """Fraction of true class 1 and all class 1"""

        return TP / (TP + FN)

    def precision(self, TP, FP):
        """Fraction of true class 1 and predicted class 1"""

        return TP / (TP + FP)

    def F1(self, TP, FP, FN):

        return TP / (TP + 0.5*(FP + FN))

    def print_metrics(self):
        """Pretty print metric dict"""

        for key, value in self.metric_dict.items():
            print(key, ': ', value)
