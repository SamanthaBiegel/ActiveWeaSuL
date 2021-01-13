import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel
from final_model import DiscriminativeModel
from plot import PlotMixin
from visualrelation import VisualRelationDataset


class ActiveLearningPipeline(PlotMixin):
    def __init__(self,
                 df,
                 final_model=False,
                 image_dir: str = "/tmp/data/visual_genome/VG_100K",
                 it: int = 100,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 batch_size: int=20,
                 active_learning: str = "probs",
                 query_strategy: str = "margin",
                 alpha: float = 0.01,
                 beta: float = 0.1,
                 penalty_strength: float = 1e3,
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
        self.batch_size = batch_size

        self.label_model = LabelModel(df=df,
                                      n_epochs=n_epochs,
                                      lr=lr,
                                      penalty_strength=penalty_strength,
                                      active_learning=active_learning,
                                      add_cliques=add_cliques,
                                      add_prob_loss=add_prob_loss,
                                      hide_progress_bar=True)
        
        self.final_model = final_model
        self.image_dir = image_dir
        self.final_probs = None

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

        self.bucket_values = abs_diff[self.unique_idx].detach().numpy()

        return self.list_options(abs_diff, minimum)

    def information_density(self):
        """Compute information density of each point"""

        I = torch.Tensor(1 / (self.N - len(self.queried)) * squareform(1 / pdist(self.X, metric="euclidean")).sum(axis=1))

        return I

    def margin_density(self, probs):
        """Query data point based on combined margin and information density measures"""

        measure = (1/self.margin(probs))**self.beta * self.information_density()**(1 - self.beta)

        maximum = max([j for i, j in enumerate(measure) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i]])

        return self.list_options(measure, maximum)[0]

    def hybrid_strategy(self, probs):

        disagreement_factor = ~((self.label_matrix.sum(axis=1) == 0) | (self.label_matrix.sum(axis=1) == self.label_matrix.shape[1]))

        abs_diff = self.margin(probs)

        minimum = min(j for i, j in enumerate(abs_diff) if disagreement_factor[i] and self.ground_truth_labels[i] == -1 and not self.all_abstain[i])

        return [j for j, v in enumerate(abs_diff) if v == minimum and disagreement_factor[j] and self.ground_truth_labels[j] == -1 and not self.all_abstain[j]]

    def relative_entropy(self, iteration):
        
        lm_posteriors = self.unique_prob_dict[iteration]
        lm_posteriors = np.concatenate([1-lm_posteriors[:, None], lm_posteriors[:, None]], axis=1).clip(1e-5, 1-1e-5)

        rel_entropy = np.zeros(len(lm_posteriors))
        sample_posteriors = np.zeros(lm_posteriors.shape)

        for i in range(len(lm_posteriors)):
            bucket_items = self.ground_truth_labels[np.where(self.unique_inverse == i)[0]]
            bucket_gt = bucket_items[bucket_items != -1]
            bucket_gt = np.array(list(bucket_gt) + [np.round(self.unique_prob_dict[0][i])])
            # if bucket_gt.size == 0:
            #     eps = 1e-2
            #     sample_posteriors[i, 1] = np.argmax(lm_posteriors[i, :]).clip(eps, 1-eps)
                
            # else:
            eps = 1e-2/(len(bucket_gt))
            sample_posteriors[i, 1] = bucket_gt.mean().clip(eps, 1-eps)

            sample_posteriors[i, 0] = 1 - sample_posteriors[i, 1]
            
            rel_entropy[i] = entropy(lm_posteriors[i, :], sample_posteriors[i, :])#/len(bucket_gt)

            if -1 not in list(bucket_items):
                rel_entropy[i] = 0

        max_buckets = np.where(rel_entropy == np.max(rel_entropy))[0]
        
        random.seed(None)
        pick_bucket = random.choice(max_buckets)

        self.bucket_values = rel_entropy

        return np.where((self.unique_inverse == pick_bucket) & (self.ground_truth_labels == -1) &~ self.all_abstain)[0]

    def query(self, probs, iteration):
        """Choose data point to label from label predictions"""
        
        random.seed(None)
        pick = random.uniform(0, 1)

        if pick < self.randomness:
            indices = [i for i in range(self.label_model.N) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i]]

        elif self.query_strategy == "margin":
            indices = self.margin_strategy(probs)
        
        elif self.query_strategy == "margin_density":
            return self.margin_density(probs)

        elif self.query_strategy == "relative_entropy":
            indices = self.relative_entropy(iteration)

        elif self.query_strategy == "hybrid":
            indices = self.hybrid_strategy(probs)

        else:
            logging.warning("Provided active learning strategy not valid, setting to margin")
            self.query_strategy = "margin"
            return self.query(probs, iteration)

        random.seed(None)

        # Pick a random point from least confident data points
        if self.add_neighbors:
            return random.sample(indices, self.add_neighbors)
        else:
            return random.choice(indices)

    def update_parameters(self, n_queried, alpha):
        """Use update rule to adjust parameters based on sampled data points"""

        if n_queried == 1:
            self.mu_0 = self.mu.clone()

        psi, _ = self._get_psi(self.label_matrix[self.ground_truth_labels != -1], self.cliques, self.nr_wl)
        mu_samples = torch.Tensor(psi[self.y[self.ground_truth_labels != -1] == 1].sum(axis=0) / n_queried)[:, None]
        self.mu = self.mu_0*np.exp(-self.alpha*n_queried) + mu_samples*(1 - np.exp(-self.alpha*n_queried))

        return self.predict(self)

    def refine_probabilities(self, label_matrix, cliques, class_balance, label_matrix_test, y_test, dl_train=None, dl_test=None):
        """Iteratively refine label matrix and training set predictions with active learning strategy"""

        self.label_matrix = label_matrix
        self.ground_truth_labels = np.full_like(self.df["y"].values, -1)
        if "x1" in self.df.columns:
            self.X = self.df[["x1", "x2"]].values
        self.y = self.df["y"].values
        self.y_test = y_test
        nr_wl = label_matrix.shape[1]
        self.all_abstain = (label_matrix == -1).sum(axis=1) == nr_wl

        if self.active_learning == "cov":
            # self.add_cliques = False
            self.label_matrix = np.concatenate([self.label_matrix, self.ground_truth_labels[:, None]], axis=1)
            label_matrix_test = np.concatenate([label_matrix_test, np.full_like(y_test, -1)[:, None]], axis=1)
            nr_wl += 1

        # if self.add_neighbors:
            # neigh = NearestNeighbors(n_neighbors=self.add_neighbors)
            # neigh.fit(self.X)

        old_probs = self.label_model.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()
        psi_test, _ = self.label_model._get_psi(label_matrix_test, cliques, nr_wl)
        test_probs = self.label_model._predict(label_matrix_test, psi_test, self.label_model.mu, torch.tensor(self.label_model.E_S))
        
        if not not self.final_model:
            if self.final_model.__class__.__name__ == "VisualRelationClassifier":
                dataset = VisualRelationDataset(image_dir=self.image_dir, 
                        df=self.df,
                        Y=old_probs.clone().detach().numpy())

                dl = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

                self.final_model = self.final_model.fit(dl)
                final_probs = self.final_model._predict(dl_train)
                final_test_probs = self.final_model._predict(dl_test)
            else:
                final_probs = self.final_model.fit(features=self.X, labels=old_probs.detach().numpy()).predict()
                final_test_probs = self.final_model.predict()
        else:
            final_probs = None
            final_test_probs = None

        _, self.unique_idx, self.unique_inverse = np.unique(old_probs.clone().detach().numpy()[:, 1], return_index=True, return_inverse=True)
        self.confs = {range(len(self.unique_idx))[i]: "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}                
        
        self.log(count=0, probs=old_probs, test_probs=test_probs, final_probs=final_probs, final_test_probs=final_test_probs)
        
        for i in tqdm(range(self.it), desc="Active Learning Iterations"):
            sel_idx = self.query(old_probs, i)
            self.ground_truth_labels[sel_idx] = self.y[sel_idx]

            if self.active_learning == "cov":
                if self.add_neighbors:
                    # neighbors_sel_idx = neigh.kneighbors(self.X[sel_idx, :][None, :], return_distance=False)
                    # self.label_matrix[neighbors_sel_idx, nr_wl] = self.y[sel_idx]
                    pass
                else:
                    self.label_matrix[sel_idx, nr_wl-1] = self.y[sel_idx]

            if not self.active_learning:
                self.label_matrix[sel_idx, :] = self.y[sel_idx]

            if self.active_learning == "update_params":
                new_probs = self.update_parameters(n_queried=i+1, alpha=self.alpha)
            else:
                # Fit label model
                new_probs = self.label_model.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

            test_probs = self.label_model._predict(label_matrix_test, psi_test, self.label_model.mu, torch.tensor(self.label_model.E_S))

            if not not self.final_model and (i+1) % 5 == 0:
                if self.final_model.__class__.__name__ == "VisualRelationClassifier":
                    dataset.Y = new_probs.clone().detach().numpy()

                    dl = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

                    self.final_model = self.final_model.fit(dl)
                    final_probs = self.final_model._predict(dl_train)
                    final_test_probs = self.final_model._predict(dl_test)
                else:
                    final_probs = self.final_model.fit(features=self.X, labels=new_probs.detach().numpy()).predict()
                    final_test_probs = self.final_model.predict()
            else:
                final_probs = None
                final_test_probs = None

            self.log(count=i+1, probs=new_probs, test_probs=test_probs, final_probs=final_probs, final_test_probs=final_test_probs, selected_point=sel_idx)

            old_probs = new_probs.clone()

            # if i % 5 == 0:
            #     self.plot_metrics()

            # if self.active_learning == "cov":
            #     if i == self.it - 1:
            #         break

            #     self.label_matrix = np.concatenate([self.label_matrix, np.full_like(self.df["y"].values, -1)[:,None]], axis=1)
            #     cliques.append([nr_wl + i + 1])

        return new_probs

    def log(self, count, probs, test_probs, final_probs, final_test_probs, selected_point=None):
        """Keep track of accuracy and other metrics"""

        if count == 0:
            self.metrics = {}
            self.test_metrics = {}
            self.final_metrics = {}
            self.final_test_metrics = {}
            self.queried = []
            self.prob_dict = {}
            self.final_prob_dict = {}
            self.unique_prob_dict = {}
            self.mu_dict = {}
            self.bucket_AL_values = {}

        self.label_model.analyze()
        self.metrics[count] = self.label_model.metric_dict
        self.test_metrics[count] = self.label_model._analyze(test_probs, self.y_test)
        self.prob_dict[count] = probs[:, 1].clone().detach().numpy()
        self.unique_prob_dict[count] = self.prob_dict[count][self.unique_idx]
        self.mu_dict[count] = self.label_model.mu.clone().detach().numpy().squeeze()

        if not not self.final_model and count % 5 == 0:
            # self.final_model.analyze()
            self.final_metrics[count] = self.final_model._analyze(final_probs, self.y)
            self.final_test_metrics[count] = self.final_model._analyze(final_test_probs, self.y_test)
            self.final_prob_dict[count] = final_probs[:, 1].clone().cpu().detach().numpy()

        if selected_point is not None:
            self.queried.append(selected_point)
            if self.query_strategy in ["relative_entropy", "margin"] and self.randomness == 0 :
                self.bucket_AL_values[count] = self.bucket_values

        return self
