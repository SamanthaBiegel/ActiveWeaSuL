import numpy as np
import torch


class PerformanceMixin:
    """Mixin that adds analysis functionality to models for computing performance measures"""

    def analyze(self, y, preds=None):
        """Get predictions and analyze performance"""
        if preds is None:
            preds = self.preds
        predicted_labels = torch.argmax(preds, dim=1).cpu().detach().numpy()
        TN, FN, FP, TP = (
            (
                (predicted_labels == i) & (y == j)
            ).sum() for i in [0, 1] for j in [0, 1]
        )
        return {
            "MCC": self.MCC(TP, TN, FP, FN),
            "Precision": self.precision(TP, FP),
            "Recall": self.recall(TP, FN),
            "Accuracy": self.accuracy(y, preds),
            "F1": self.F1(TP, FP, FN)
        }

    def accuracy(self, y, preds=None):
        """Compute overall accuracy from label predictions"""
        if preds is None:
            preds = self.preds
        return np.array(
            torch
            .argmax(preds, dim=1)
            .cpu()
            .detach()
            .numpy() == y
        ).sum() / len(y)

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
        return TP / (TP + 0.5 * (FP + FN))
