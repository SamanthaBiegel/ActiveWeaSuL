import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel
from final_model import DiscriminativeModel
from plot import PlotMixin


class ActiveLearningPipeline(PlotMixin):
    def __init__(self,
                 df,
                 final_model=False,
                 it: int = 100,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 active_learning: str = "probs",
                 query_strategy: str = "margin",
                 alpha: float = 0.01,
                 beta: float = 0.1,
                 add_neighbors: int = 0,
                 add_cliques: bool = True,
                 add_prob_loss: bool = False,
                 randomness: float = 0):

        self.it = it
        self.query_strategy = query_strategy
        self.alpha = alpha
        self.beta = beta
        self.add_neighbors = add_neighbors
        self.randomness = randomness
        self.df = df
        self.active_learning = active_learning

        self.label_model = LabelModel(df=df,
                                      n_epochs=n_epochs,
                                      lr=lr,
                                      active_learning=active_learning,
                                      add_cliques=add_cliques,
                                      add_prob_loss=add_prob_loss,
                                      hide_progress_bar=True)
        
        self.final_model = final_model
        self.final_probs = None

    def entropy(self, probs):

        prod = probs * torch.log(probs)
        prod[torch.isnan(prod)] = 0

        return - prod.sum(axis=1)

    def list_options(self, values, criterium):
        """Return options with equal informativeness according to criterium"""

        return [j for j, v in enumerate(values) if v == criterium and self.ground_truth_labels[j] == -1 and not self.all_abstain[j]]

    def margin(self, probs):
        """P(Y=1|...) - P(Y=0|...)"""

        abs_diff = torch.abs(probs[:, 1] - probs[:, 0])

        return abs_diff

    def margin_strategy(self, probs):
        """List query options based on minimum margin strategy"""

        abs_diff = self.margin(probs)

        minimum = min(j for i, j in enumerate(abs_diff) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i])

        return self.list_options(abs_diff, minimum)

    def uncertainty(self, probs):
        """1 - P(Y=1|...)"""

        return 1 - torch.max(probs, dim=1, keepdim=True).values

    def entropy_strategy(self, probs):
        """List query options based on maximum entropy strategy"""

        H = self.entropy(probs)

        maximum = max(j for i, j in enumerate(H) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i])

        return self.list_options(H, maximum)

    def information_density(self):
        """Compute information density of each point"""

        I = torch.Tensor(1 / (self.N - len(self.queried)) * squareform(1 / pdist(self.X, metric="euclidean")).sum(axis=1))

        return I

    def margin_density(self, probs):
        """Query data point based on combined margin and information density measures"""

        measure = (1/self.margin(probs))**self.beta * self.information_density()**(1 - self.beta)

        maximum = max([j for i, j in enumerate(measure) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i]])

        return self.list_options(measure, maximum)[0]

    def query(self, probs, iteration):
        """Choose data point to label from label predictions"""

        pick = np.random.uniform()

        if pick < self.randomness:
            indices = [i for i in range(self.N) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i]]

        elif self.query_strategy == "margin":
            indices = self.margin_strategy(probs)

        elif self.query_strategy == "entropy":
            indices = self.entropy_strategy(probs)
        
        elif self.query_strategy == "margin_density":
            return self.margin_density(probs)

        elif self.query_strategy == "test":
            indices = self.test_strategy(iteration)

        # Make really random
        random.seed(random.SystemRandom().random())

        # Pick a random point from least confident data points
        if self.add_neighbors:
            return random.sample(indices, self.add_neighbors)
        else:
            return random.choice(indices)

    def logging(self, count, probs, final_probs=None, selected_point=None):
        """Keep track of accuracy and other metrics"""

        if count == 0:
            self.metrics = {}
            self.final_metrics = {}
            self.queried = []
            self.prob_dict = {}
            self.final_prob_dict = {}
            self.unique_prob_dict = {}
            self.mu_dict = {}

        self.label_model.analyze()
        self.metrics[count] = self.label_model.metric_dict
        self.prob_dict[count] = probs[:, 1].clone().detach().numpy()
        self.unique_prob_dict[count] = self.prob_dict[count][self.unique_idx]
        self.mu_dict[count] = self.label_model.mu.clone().detach().numpy().squeeze()

        if not not self.final_model:
            self.final_model.analyze()
            self.final_metrics[count] = self.final_model.metric_dict
            self.final_prob_dict[count] = final_probs[:, 1]

        if selected_point is not None:
            self.queried.append(selected_point)

        return self

    def update_parameters(self, n_queried, alpha):
        """Use update rule to adjust parameters based on sampled data points"""

        if n_queried == 1:
            self.mu_0 = self.mu.clone()

        psi, _ = self._get_psi(self.label_matrix[self.ground_truth_labels != -1], self.cliques, self.nr_wl)
        mu_samples = torch.Tensor(psi[self.y[self.ground_truth_labels != -1] == 1].sum(axis=0) / n_queried)[:, None]
        self.mu = self.mu_0*np.exp(-self.alpha*n_queried) + mu_samples*(1 - np.exp(-self.alpha*n_queried))

        return self.predict(self)

    def test_strategy(self, iteration):

        if iteration == 0:
            return [i for i in range(self.label_model.N) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i]]
        else:
            diff_prob_labels = self.prob_dict[iteration] - self.prob_dict[iteration-1]
            return np.where(self.unique_inverse == self.unique_inverse[np.argmax(diff_prob_labels)])[0]

    def refine_probabilities(self, label_matrix, cliques, class_balance):
        """Iteratively refine label matrix and training set predictions with active learning strategy"""

        self.label_matrix = label_matrix
        self.ground_truth_labels = np.full_like(self.df["y"].values, -1) 
        # self.X = self.df[["x1", "x2"]].values
        self.y = self.df["y"].values
        nr_wl = label_matrix.shape[1]
        self.all_abstain = (label_matrix == -1).sum(axis=1) == nr_wl

        if self.active_learning == "cov":
            # self.add_cliques = False
            self.label_matrix = np.concatenate([self.label_matrix, self.ground_truth_labels[:, None]], axis=1)

        # if self.add_neighbors:
            # neigh = NearestNeighbors(n_neighbors=self.add_neighbors)
            # neigh.fit(self.X)

        old_probs = self.label_model.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()
        if not not self.final_model:
            self.final_probs = self.final_model.fit(features=self.X, labels=old_probs.detach().numpy()).predict()

        _, self.unique_idx, self.unique_inverse = np.unique(old_probs.clone().detach().numpy()[:, 1], return_index=True, return_inverse=True)

        self.logging(count=0, final_probs=self.final_probs, probs=old_probs)

        for i in tqdm(range(self.it)):
            sel_idx = self.query(old_probs, i)
            self.ground_truth_labels[sel_idx] = self.y[sel_idx]

            if self.active_learning == "cov":
                if self.add_neighbors:
                    # neighbors_sel_idx = neigh.kneighbors(self.X[sel_idx, :][None, :], return_distance=False)
                    # self.label_matrix[neighbors_sel_idx, nr_wl] = self.y[sel_idx]
                    pass
                else:
                    self.label_matrix[sel_idx, nr_wl] = self.y[sel_idx]

            if self.active_learning == "update_params":
                new_probs = self.update_parameters(n_queried=i+1, alpha=self.alpha)
            else:
                # Fit label model
                new_probs = self.label_model.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

            if not not self.final_model:
                self.final_probs = self.final_model.fit(features=self.X, labels=new_probs.detach().numpy()).predict()
            self.logging(count=i+1, probs=new_probs, final_probs=self.final_probs, selected_point=sel_idx)

            old_probs = new_probs.clone()

            # if self.active_learning == "cov":
            #     if i == self.it - 1:
            #         break

            #     self.label_matrix = np.concatenate([self.label_matrix, np.full_like(self.df["y"].values, -1)[:,None]], axis=1)
            #     cliques.append([nr_wl + i + 1])

        self.confs = {range(len(self.unique_idx))[i]: "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}

        return new_probs
