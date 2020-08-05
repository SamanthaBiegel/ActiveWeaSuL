import numpy as np
import random
import torch
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel


class ActiveLearningPipeline(LabelModel):
    def __init__(self,
                 final_model_kwargs,
                 df,
                 it: int = 100,
                 active_learning: str = "probs",
                 add_cliques: bool = True,
                 add_prob_loss: bool = False):

        self.it = it
        super().__init__(final_model_kwargs=final_model_kwargs,
                         df=df,
                         active_learning=active_learning,
                         add_cliques=add_cliques,
                         add_prob_loss=add_prob_loss,
                         hide_progress_bar=True)

    def query(self, probs):
        """Choose data point to label from label predictions"""

        abs_diff = torch.abs(probs[:, 1] - probs[:, 0])

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

    def refine_probabilities(self, label_matrix, cliques, class_balance):
        """Iteratively refine label matrix and training set predictions with active learning strategy"""

        self.accuracies = []
        self.queried = []
        self.prob_dict = {}
        self.unique_prob_dict = {}

        self.label_matrix = label_matrix

        if self.active_learning == "cov":
            # self.add_cliques = False
            self.ground_truth_labels = None
        if self.active_learning == "probs":
            self.add_cliques = True
            self.ground_truth_labels = np.full_like(self.df["y"].values, -1)

        old_probs = self.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

        _, unique_idx = np.unique(old_probs.clone().detach().numpy()[:, 1], return_index=True)

        self.accuracies.append(self.accuracy())
        self.prob_dict[0] = old_probs[:, 1].clone().detach().numpy()
        self.unique_prob_dict[0] = self.prob_dict[0][unique_idx]

        for i in tqdm(range(self.it)):
            sel_idx = self.query(old_probs)

            # print("Iteration:", i + 1, " Label combination", label_matrix[sel_idx, :nr_wl], " True label:", y[sel_idx],
            #       "Estimated label:", Y_hat[sel_idx], " selected index:", sel_idx)
           
            if self.active_learning == "probs":
                self.ground_truth_labels[sel_idx] = self.df["y"].values[sel_idx]
            if self.active_learning == "cov":
                self.label_matrix[sel_idx, self.nr_wl - 1] = self.df["y"].values[sel_idx]

            # Fit label model on refined label matrix
            new_probs = self.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

            # print("Before:", old_probs[sel_idx, :], "After:", new_probs[sel_idx])

            self.accuracies.append(self.accuracy())
            self.queried.append(sel_idx)
            self.prob_dict[i+1] = new_probs[:, 1].clone().detach().numpy()
            self.unique_prob_dict[i+1] = self.prob_dict[i+1][unique_idx]

            old_probs = new_probs.clone()

        return new_probs
