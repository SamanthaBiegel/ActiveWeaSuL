import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel, get_overall_accuracy


class ActiveLearningPipeline(LabelModel):
    def __init__(self, it, df, final_model_kwargs):
        self.it = it
        super().__init__(final_model_kwargs, df)

    def query(self, probs):
        """Choose data point to label from label predictions"""

        abs_diff = np.abs(probs[:, 1] - probs[:, 0])

        # Find data points the model where the model is least confident
        if self.active_learning == "probs":
            minimum = min(j for i, j in enumerate(abs_diff) if self.ground_truth_labels[i] == -1)
        if self.active_learning == "cov":
            minimum = min(j for i, j in enumerate(abs_diff) if self.label_matrix[i, self.nr_wl - 1] == -1)
        indices = [j for j, v in enumerate(abs_diff) if v == minimum]

        # Make really random
        random.seed(random.SystemRandom().random())

        # Pick a random point from least confident data points
        return random.choice(indices)

    def refine_probabilities(self, label_matrix, cliques, class_balance, active_learning):
        """Iteratively refine label matrix and training set predictions with active learning strategy"""

        self.accuracies = []
        self.queried = []

        self.label_matrix = label_matrix
        self.active_learning = active_learning

        if self.active_learning == "cov":
            self.add_cliques = False
            self.ground_truth_labels = None
        if self.active_learning == "probs":
            self.add_cliques = True
            self.ground_truth_labels = np.full_like(self.df["y"].values, -1)

        lm = LabelModel(final_model_kwargs=self.final_model_kwargs, df=self.df, active_learning=self.active_learning, add_cliques=self.add_cliques)

        old_probs = lm.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

        self.accuracies.append(lm.accuracy())

        for i in range(self.it):
            sel_idx = self.query(old_probs)

            # print("Iteration:", i + 1, " Label combination", label_matrix[sel_idx, :nr_wl], " True label:", y[sel_idx],
            #       "Estimated label:", Y_hat[sel_idx], " selected index:", sel_idx)

            # Add label to label matrix
            # label_matrix[sel_idx, nr_wl - 1] = y[sel_idx]
            self.ground_truth_labels[sel_idx] = self.df["y"].values[sel_idx]

            # Fit label model on refined label matrix
            new_probs = lm.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

            # print("Before:", old_probs[sel_idx, :], "After:", new_probs[sel_idx])

            self.accuracies.append(lm.accuracy())
            self.queried.append(sel_idx)

            old_probs = new_probs.copy()

        return new_probs
