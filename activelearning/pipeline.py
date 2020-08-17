import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel


class ActiveLearningPipeline(LabelModel):
    def __init__(self,
                 final_model_kwargs,
                 df,
                 it: int = 100,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 active_learning: str = "probs",
                 query_strategy: str = "margin",
                 alpha: float = 0.01,
                 add_neighbors: int = 0,
                 add_cliques: bool = True,
                 add_prob_loss: bool = False,
                 randomness: float = 0):

        self.it = it
        self.query_strategy = query_strategy
        self.alpha = alpha
        self.add_neighbors = add_neighbors
        self.randomness = randomness
        super().__init__(final_model_kwargs=final_model_kwargs,
                         df=df,
                         n_epochs=n_epochs,
                         lr=lr,
                         active_learning=active_learning,
                         add_cliques=add_cliques,
                         add_prob_loss=add_prob_loss,
                         hide_progress_bar=True)

    def entropy(self, probs):
    
        prod = probs * torch.log(probs)
        prod[torch.isnan(prod)] = 0
        
        return - prod.sum(axis=1)

    def list_options(self, values, criterium):

        return [j for j, v in enumerate(values) if v == criterium and self.ground_truth_labels[j] == -1]

    def margin_strategy(self, probs):

        abs_diff = torch.abs(probs[:, 1] - probs[:, 0])

        # Find data points the model where the model is least confident
        minimum = min(j for i, j in enumerate(abs_diff) if self.ground_truth_labels[i] == -1)
        
        return self.list_options(abs_diff, minimum)

    def entropy_strategy(self, probs):

        H = self.entropy(probs)

        maximum = max(j for i, j in enumerate(H) if self.ground_truth_labels[i] == -1)

        return self.list_options(H, maximum)

    def query(self, probs):
        """Choose data point to label from label predictions"""

        pick = np.random.uniform()

        if pick < self.randomness:
            indices = [i for i in range(self.N) if self.ground_truth_labels[i] == -1]

        elif self.query_strategy == "margin":
            indices = self.margin_strategy(probs)

        elif self.query_strategy == "entropy":
            indices = self.entropy_strategy(probs)

        # Make really random
        random.seed(random.SystemRandom().random())

        # Pick a random point from least confident data points
        if self.add_neighbors:
            return random.sample(indices, self.add_neighbors)
        else:
            return random.choice(indices)

    def logging(self, count, probs, selected_point=None):

        if count == 0:
            self.accuracies = []
            self.queried = []
            self.prob_dict = {}
            self.unique_prob_dict = {}
            self.mu_dict = {}

        self.accuracies.append(self._accuracy(probs, self.y))
        self.prob_dict[count] = probs[:, 1].clone().detach().numpy()
        self.unique_prob_dict[count] = self.prob_dict[count][self.unique_idx]
        self.mu_dict[count] = self.mu.clone().detach().numpy().squeeze()

        if selected_point is not None:
            self.queried.append(selected_point)

        return self

    def update_parameters(self, n_queried, alpha):

        if n_queried == 1:
            self.mus = self.mu

        psi, _ = self._get_psi(self.label_matrix[self.ground_truth_labels != -1])
        mu_samples = torch.Tensor(psi[self.y[self.ground_truth_labels != -1] == 1].sum(axis=0) / n_queried)[:, None]
        mu_updated = self.mu*np.exp(-self.alpha*n_queried) + mu_samples*(1 - np.exp(-self.alpha*n_queried))

        self.mus = torch.cat((self.mus, mu_updated), axis=1)

        return self._predict(mu_updated, self.E_S)

    def refine_probabilities(self, label_matrix, cliques, class_balance):
        """Iteratively refine label matrix and training set predictions with active learning strategy"""

        self.label_matrix = label_matrix
        self.ground_truth_labels = np.full_like(self.df["y"].values, -1)
        X = self.df[["x1", "x2"]].values
        self.y = self.df["y"].values
        nr_wl = label_matrix.shape[1]

        if self.active_learning == "cov":
            # self.add_cliques = False
            self.label_matrix = np.concatenate([self.label_matrix, self.ground_truth_labels], axis=1)

        if self.add_neighbors:
            neigh = NearestNeighbors(n_neighbors=self.add_neighbors)
            neigh.fit(X)

        old_probs = self.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

        _, self.unique_idx = np.unique(old_probs.clone().detach().numpy()[:, 1], return_index=True)

        self.logging(count=0, probs=old_probs)

        for i in tqdm(range(self.it)):
            sel_idx = self.query(old_probs)
            self.ground_truth_labels[sel_idx] = self.y[sel_idx]

            if self.active_learning == "cov":
                if self.add_neighbors:
                    neighbors_sel_idx = neigh.kneighbors(X[sel_idx, :][None, :], return_distance=False)
                    self.label_matrix[neighbors_sel_idx, nr_wl] = self.y[sel_idx]
                else:
                    self.label_matrix[sel_idx, nr_wl] = self.y[sel_idx]

            if self.active_learning == "update_params":
                new_probs = self.update_parameters(n_queried=i+1, alpha=self.alpha)
            else:
                # Fit label model on refined label matrix
                new_probs = self.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

            self.logging(count=i+1, probs=new_probs, selected_point=sel_idx)

            old_probs = new_probs.clone()

            # if self.active_learning == "cov":
            #     if i == self.it - 1:
            #         break

            #     self.label_matrix = np.concatenate([self.label_matrix, np.full_like(self.df["y"].values, -1)[:,None]], axis=1)
            #     cliques.append([nr_wl + i + 1])

        return new_probs
