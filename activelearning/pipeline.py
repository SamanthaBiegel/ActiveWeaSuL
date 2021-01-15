import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel
from final_model import DiscriminativeModel
from plot import PlotMixin
from visualrelation import VisualRelationDataset


class ActiveLearningPipeline(PlotMixin):
    def __init__(self,
                 y_true,
                 it: int = 30,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 penalty_strength: float = 1e3,
                 query_strategy: str = "maxkl",
                 randomness: float = 0,
                 final_model=False,
                 df=None,
                 image_dir: str = "/tmp/data/visual_genome/VG_100K",
                 batch_size: int = 20,
                 discr_model_frequency: int = 1):
        """Pipeline to run Active WeaSuL

        Args:
            y_true (numpy.array): Ground truth labels of training dataset
            it (int, optional): Number of active learning iterations
            n_epochs (int, optional): Number of label model epochs
            lr (float, optional): Label model learning rate
            penalty_strength (float, optional): Strength of the active learning penalty
            query_strategy (str, optional): Active learning query strategy, one of ["maxkl", "margin", "nashaat"]
            randomness (float, optional): Probability of choosing a random point instead of following strategy
            final_model: Optional discriminative model object
            df (pandas.DataFrame, optional): Dataframe that contains discriminative model input (features)
            image_dir (str, optional): Directory of images if training discriminative model on image data
            batch_size (int, optional): Batch size if training discriminate model on image data
            discr_model_frequency (int, optional): How often to train discriminative model (interval)
        """
        self.y_true = y_true
        self.it = it
        self.label_model = LabelModel(y_true=y_true,
                                      n_epochs=n_epochs,
                                      lr=lr,
                                      hide_progress_bar=True)
        self.penalty_strength = penalty_strength
        self.query_strategy = query_strategy
        self.randomness = randomness
        self.final_model = final_model
        self.df = df
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.discr_model_frequency = discr_model_frequency

    def margin(self, probs):
        """P(Y=1|...) - P(Y=0|...)"""

        abs_diff = torch.abs(probs[:, 1] - probs[:, 0])

        return abs_diff

    def margin_strategy(self, probs):
        """Choose points to sample from based on uncertainty of probabilistic label

        Args:
            probs (numpy.array): Array with probabilistic labels for training dataset

        Returns:
            list: List of indices of unlabeled points to choose from
        """

        # Find highest uncertainty of probabilistic label in pool
        abs_diff = self.margin(probs)
        minimum = min(j for i, j in enumerate(abs_diff) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i])

        self.bucket_values = abs_diff[self.unique_idx].detach().numpy()

        # Select points with highest uncertainty
        return [j for j, v in enumerate(abs_diff) if v == minimum and self.ground_truth_labels[j] == -1 and not self.all_abstain[j]]

    def nashaat_strategy(self, probs):
        """Implement approach by Nashaat et al.

        Args:
            probs (numpy.array): Array with probabilistic labels for training dataset

        Returns:
            list: List of indices of unlabeled points to choose from
        """

        # Identify pool of points with labeling function conflicts
        disagreement_factor = ~((self.label_matrix.sum(axis=1) == 0) | (self.label_matrix.sum(axis=1) == self.label_matrix.shape[1]))

        # Find highest uncertainty of probabilistic label in pool
        abs_diff = self.margin(probs)
        minimum = min(j for i, j in enumerate(abs_diff) if disagreement_factor[i] and self.ground_truth_labels[i] == -1 and not self.all_abstain[i])

        # Select points with highest uncertainty
        return [j for j, v in enumerate(abs_diff) if v == minimum and disagreement_factor[j] and self.ground_truth_labels[j] == -1 and not self.all_abstain[j]]

    def maxkl_strategy(self, iteration):
        """Choose bucket of points to sample from following MaxKL query strategy

        Args:
            iteration (int): Current iteration number

        Returns:
            numpy.array: Array of indices of unlabeled points in chosen bucket
        """
        
        # Label model distributions
        lm_posteriors = self.unique_prob_dict[iteration]
        lm_posteriors = np.concatenate([1-lm_posteriors[:, None], lm_posteriors[:, None]], axis=1).clip(1e-5, 1-1e-5)

        # Sample distributions
        # D_KL(LM distribution||Sample distribution)
        rel_entropy = np.zeros(len(lm_posteriors))
        sample_posteriors = np.zeros(lm_posteriors.shape)
        # Iterate over buckets
        for i in range(len(lm_posteriors)):
            # Collect points in bucket
            bucket_items = self.ground_truth_labels[np.where(self.unique_inverse == i)[0]]
            # Collect labeled points in bucket
            bucket_gt = bucket_items[bucket_items != -1]
            # Add initial labeled point
            bucket_gt = np.array(list(bucket_gt) + [np.round(self.unique_prob_dict[0][i])])

            # Bucket distribution, clip to avoid D_KL undefined
            eps = 1e-2/(len(bucket_gt))
            sample_posteriors[i, 1] = bucket_gt.mean().clip(eps, 1-eps)
            sample_posteriors[i, 0] = 1 - sample_posteriors[i, 1]

            # KL divergence
            rel_entropy[i] = entropy(lm_posteriors[i, :], sample_posteriors[i, :])#/len(bucket_gt)

            # If no unlabeled points left, stop considering bucket
            if -1 not in list(bucket_items):
                rel_entropy[i] = 0

            bucket_labels = self.y_true[np.where(self.unique_inverse == i)[0]][bucket_items == -1]
            if (1 not in list(bucket_labels)) and (0 not in list(bucket_labels)):
                rel_entropy[i] = 0

        self.bucket_values = rel_entropy

        # Pick bucket 
        max_buckets = np.where(rel_entropy == np.max(rel_entropy))[0]
        random.seed(None)
        pick_bucket = random.choice(max_buckets)

        # Pick point from bucket
        return np.where((self.unique_inverse == pick_bucket) & (self.ground_truth_labels == -1) & ~self.all_abstain & (self.y_true != -1))[0]

    def query(self, probs, iteration):
        """Choose data point to label following query strategy

        Args:
            probs (numpy.array): Array with probabilistic labels for training dataset
            iteration (int): Current iteration number

        Returns:
            int: Index of chosen point
        """
        
        random.seed(None)
        pick = random.uniform(0, 1)

        # Choose random point instead of following strategy
        if pick < self.randomness:
            indices = [i for i in range(self.label_model.N) if self.ground_truth_labels[i] == -1 and not self.all_abstain[i]]

        elif self.query_strategy == "margin":
            indices = self.margin_strategy(probs)

        elif self.query_strategy == "maxkl":
            indices = self.maxkl_strategy(iteration)

        elif self.query_strategy == "nashaat":
            indices = self.nashaat_strategy(probs)

        else:
            logging.warning("Provided active learning strategy not valid, setting to maxkl")
            self.query_strategy = "maxkl"
            return self.query(probs, iteration)

        random.seed(None)
        # Pick a random point from selected subset
        return random.choice(indices)

    def run_active_learning(self, label_matrix, cliques, class_balance, label_matrix_test, y_test, dl_train=None, dl_test=None):
        """Iteratively label points, refit label model and return adjusted probabilistic labels

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on train set
            cliques (list): List of lists of maximal cliques (column indices of label matrix)
            class_balance (numpy.array): Array with true class distribution
            label_matrix_test (numpy.array): Array with labeling function outputs on test set
            y_test (numpy.array): Ground truth labels of test set
            dl_train (torch.utils.data.DataLoader, optional): Train dataloader if training discriminative model on image data
            dl_test (torch.utils.data.DataLoader, optional): Test dataloader if training discriminative model on image data

        Returns:
            np.array: Array with probabilistic labels for training dataset
        """

        self.label_matrix = label_matrix
        self.ground_truth_labels = np.full_like(self.y_true, -1)
        self.y_test = y_test
        nr_wl = label_matrix.shape[1]
        self.all_abstain = (label_matrix == -1).sum(axis=1) == nr_wl

        # Initial fit and predict label model
        prob_labels_train = self.label_model.fit(label_matrix=self.label_matrix,
                                                 cliques=cliques,
                                                 class_balance=class_balance).predict()
        prob_labels_test = self.label_model._predict(label_matrix_test,
                                                     self.label_model.mu,
                                                     torch.tensor(self.label_model.E_S))
        
        # Optionally, train discriminative model on probabilistic labels
        if not not self.final_model:
            if self.final_model.__class__.__name__ == "VisualRelationClassifier":
                # VRD/VG dataset
                dataset = VisualRelationDataset(image_dir=self.image_dir,
                                                df=self.df,
                                                Y=prob_labels_train.clone().detach().numpy())
                dl = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

                self.final_model = self.final_model.fit(dl)
                preds_train = self.final_model._predict(dl_train)
                preds_test = self.final_model._predict(dl_test)
            else:
                # Synthetic dataset
                self.X = self.df[["x1", "x2"]].values
                preds_train = self.final_model.fit(features=self.X, labels=prob_labels_train.detach().numpy()).predict()
                preds_test = self.final_model.predict()
        else:
            preds_train = None
            preds_test = None

        # Identify buckets
        _, self.unique_idx, self.unique_inverse = np.unique(prob_labels_train.clone().detach().numpy()[:, 1],
                                                            return_index=True,
                                                            return_inverse=True)
        self.confs = {range(len(self.unique_idx))[i]:
                      "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}
        
        self.log(count=0, probs=prob_labels_train, test_probs=prob_labels_test, final_probs=preds_train, final_test_probs=preds_test)
        
        for i in tqdm(range(self.it), desc="Active Learning Iterations"):
            # Switch to active learning mode
            self.label_model.active_learning = True
            self.label_model.penalty_strength = self.penalty_strength

            # Query point and add to ground truth labels
            sel_idx = self.query(prob_labels_train, i)
            self.ground_truth_labels[sel_idx] = self.y_true[sel_idx]
            if self.query_strategy == "nashaat":
                # Nashaat et al. replace labeling function outputs by ground truth
                self.label_matrix[sel_idx, :] = self.y_true[sel_idx]

            # Fit label model with penalty and predict
            prob_labels_train = self.label_model.fit(label_matrix=self.label_matrix,
                                                     cliques=cliques,
                                                     class_balance=class_balance,
                                                     ground_truth_labels=self.ground_truth_labels).predict()
            prob_labels_test = self.label_model._predict(label_matrix_test,
                                                         self.label_model.mu,
                                                         torch.tensor(self.label_model.E_S))

            # Optionally, train discriminative model on probabilistic labels
            if not not self.final_model and (i+1) % self.discr_model_frequency == 0:
                if self.final_model.__class__.__name__ == "VisualRelationClassifier":
                    # VRD/VG dataset
                    dataset.Y = prob_labels_train.clone().detach().numpy()

                    dl = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

                    self.final_model = self.final_model.fit(dl)
                    preds_train = self.final_model._predict(dl_train)
                    preds_test = self.final_model._predict(dl_test)
                else:
                    # Synthetic dataset
                    preds_train = self.final_model.fit(features=self.X, 
                                                       labels=prob_labels_train.detach().numpy()).predict()
                    preds_test = self.final_model.predict()
            else:
                preds_train = None
                preds_test = None

            self.log(count=i+1, probs=prob_labels_train, test_probs=prob_labels_test, final_probs=preds_train,
                     final_test_probs=preds_test, selected_point=sel_idx)

        return prob_labels_train

    def log(self, count, probs, test_probs, final_probs, final_test_probs, selected_point=None):
        """Keep track of performance metrics and label predictions"""

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

        if not not self.final_model and count % self.discr_model_frequency == 0:
            # self.final_model.analyze()
            self.final_metrics[count] = self.final_model._analyze(final_probs, self.y_true)
            self.final_test_metrics[count] = self.final_model._analyze(final_test_probs, self.y_test)
            self.final_prob_dict[count] = final_probs[:, 1].clone().cpu().detach().numpy()

        if selected_point is not None:
            self.queried.append(selected_point)
            if self.query_strategy in ["maxkl", "margin"] and self.randomness == 0:
                self.bucket_AL_values[count] = self.bucket_values

        return self
