import logging
import numpy as np
import random
from scipy.stats import entropy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel
# from logisticregression import LogisticRegression
from plot import PlotMixin


class ActiveWeaSuLPipeline(PlotMixin):
    """Pipeline to run Active WeaSuL.

    Args:
        it (int, optional): Number of active learning iterations
        n_epochs (int, optional): Number of label model epochs
        lr (float, optional): Label model learning rate
        penalty_strength (float, optional): Strength of the active learning penalty
        query_strategy (str, optional): Active learning query strategy, one of ["maxkl", "margin", "nashaat"]
        randomness (float, optional): Probability of choosing a random point instead of following strategy
        final_model: Optional discriminative model object
        image_dir (str, optional): Directory of images if training discriminative model on image data
        batch_size (int, optional): Batch size if training discriminate model on image data
        discr_model_frequency (int, optional): How often to train discriminative model (interval)
    """

    def __init__(self,
                 it: int = 30,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 penalty_strength: float = 1e3,
                 query_strategy: str = "maxkl",
                 randomness: float = 0,
                 final_model=None,
                 image_dir: str = "/tmp/data/visual_genome/VG_100K",
                 batch_size: int = 20,
                 discr_model_frequency: int = 1):

        self.it = it
        self.label_model = LabelModel(n_epochs=n_epochs,
                                      lr=lr,
                                      hide_progress_bar=True)
        self.penalty_strength = penalty_strength
        self.query_strategy = query_strategy
        self.randomness = randomness
        self.final_model = final_model
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
        lm_posteriors = self.probs["bucket_labels_train"][iteration]
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
            bucket_gt = list(bucket_items[bucket_items != -1])
            # Add initial labeled point
            if not bucket_gt:
                bucket_gt.append(int(np.round(self.probs["bucket_labels_train"][0][i].clip(0,1))))
            bucket_gt = np.array(bucket_gt)

            # Bucket distribution, clip to avoid D_KL undefined
            eps = 1e-2/(len(bucket_gt))
            sample_posteriors[i, 1] = bucket_gt.mean().clip(eps, 1-eps)
            sample_posteriors[i, 0] = 1 - sample_posteriors[i, 1]

            # KL divergence
            rel_entropy[i] = entropy(lm_posteriors[i, :], sample_posteriors[i, :])#/len(bucket_gt)

            # If no unlabeled points left, stop considering bucket
            if -1 not in list(bucket_items):
                rel_entropy[i] = 0

            bucket_labels = self.y_train[np.where(self.unique_inverse == i)[0]][bucket_items == -1]
            if (1 not in list(bucket_labels)) and (0 not in list(bucket_labels)):
                rel_entropy[i] = 0

        self.bucket_values = rel_entropy

        # Pick bucket 
        max_buckets = np.where(rel_entropy == np.max(rel_entropy))[0]
        random.seed(None)
        pick_bucket = random.choice(max_buckets)

        # Pick point from bucket
        return np.where((self.unique_inverse == pick_bucket) & (self.ground_truth_labels == -1) & ~self.all_abstain & (self.y_train != -1))[0]

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

    def run_active_weasul(self, label_matrix, y_train, cliques, class_balance, label_matrix_test=None, y_test=None, train_dataset=None, test_dataset=None):
        """Iteratively label points, refit label model and return adjusted probabilistic labels

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on train set
            y_train (numpy.array): Ground truth labels of training dataset
            cliques (list): List of lists of maximal cliques (column indices of label matrix)
            class_balance (numpy.array): Array with true class distribution
            label_matrix_test (numpy.array): Array with labeling function outputs on test set
            y_test (numpy.array): Ground truth labels of test set
            train_dataset (torch.utils.data.Dataset, optional): Train dataset if training discriminative model on image data
            test_dataset (torch.utils.data.Dataset, optional): Test dataset if training discriminative model on image data

        Returns:
            np.array: Array with probabilistic labels for training dataset
        """

        if any(v is None for v in (label_matrix_test, y_test, test_dataset)):
            label_matrix_test = label_matrix.copy()
            y_test = y_train.copy()
            test_dataset = train_dataset

        self.label_matrix = label_matrix
        self.ground_truth_labels = np.full_like(y_train, -1)
        self.y_train = y_train
        self.y_test = y_test
        nr_wl = label_matrix.shape[1]
        self.all_abstain = (label_matrix == -1).sum(axis=1) == nr_wl

        # Identify buckets
        _, self.unique_idx, self.unique_inverse = np.unique(label_matrix,
                                                            return_index=True,
                                                            return_inverse=True,
                                                            axis=0)
        self.confs = {range(len(self.unique_idx))[i]:
                      "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}

        # Initial fit and predict label model
        prob_labels_train = self.label_model.fit(label_matrix=self.label_matrix,
                                                 cliques=cliques,
                                                 class_balance=class_balance).predict()
        prob_labels_test = self.label_model.predict(label_matrix_test,
                                                    self.label_model.mu,
                                                    self.label_model.E_S)

        # Optionally, train discriminative model on probabilistic labels
        if self.final_model is not None:
            train_dataset.Y = prob_labels_train.clone().detach()
            dl_train = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
            dl_test = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)

            preds_train = self.final_model.fit(dl_train).predict()
            preds_test = self.final_model.predict(dl_test)
        else:
            preds_train = None
            preds_test = None

        self.log(count=0, lm_train=prob_labels_train, lm_test=prob_labels_test, fm_train=preds_train, fm_test=preds_test)

        for i in tqdm(range(self.it), desc="Active Learning Iterations"):
            # Switch to active learning mode
            self.label_model.active_learning = True
            self.label_model.penalty_strength = self.penalty_strength

            # Query point and add to ground truth labels
            sel_idx = self.query(prob_labels_train, i)
            self.ground_truth_labels[sel_idx] = self.y_train[sel_idx]
            if self.query_strategy == "nashaat":
                self.label_model.active_learning = False
                # Nashaat et al. replace labeling function outputs by ground truth
                self.label_matrix[sel_idx, :] = self.y_train[sel_idx]

            # Fit label model with penalty and predict
            prob_labels_train = self.label_model.fit(label_matrix=self.label_matrix,
                                                     cliques=cliques,
                                                     class_balance=class_balance,
                                                     ground_truth_labels=self.ground_truth_labels).predict()
            prob_labels_test = self.label_model.predict(label_matrix_test,
                                                        self.label_model.mu,
                                                        self.label_model.E_S)

            # Optionally, train discriminative model on probabilistic labels
            if self.final_model is not None and (i+1) % self.discr_model_frequency == 0:
                train_dataset.Y = prob_labels_train.clone().detach()
                dl_train = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)

                preds_train = self.final_model.fit(dl_train).predict()
                preds_test = self.final_model.predict(dl_test)
            else:
                preds_train = None
                preds_test = None

            self.log(count=i+1, lm_train=prob_labels_train, lm_test=prob_labels_test, fm_train=preds_train,
                     fm_test=preds_test, selected_point=sel_idx)

        return prob_labels_train

    def log(self, count, lm_train, lm_test, fm_train, fm_test, selected_point=None):
        """Keep track of performance metrics and label predictions"""

        if count == 0:
            self.metrics = {}
            self.metrics["Generative_train"] = {}
            self.metrics["Generative_test"] = {}
            self.metrics["Discriminative_train"] = {}
            self.metrics["Discriminative_test"] = {}
            self.queried = []
            self.probs = {}
            self.probs["Generative_train"] = {}
            self.probs["Generative_test"] = {}
            self.probs["Discriminative_train"] = {}
            self.probs["Discriminative_test"] = {}
            self.probs["bucket_labels_train"] = {}
            self.mu_dict = {}
            self.bucket_AL_values = {}

        self.metrics["Generative_train"][count] = self.label_model.analyze(self.y_train)
        self.metrics["Generative_test"][count] = self.label_model.analyze(self.y_test, lm_test)
        self.probs["Generative_train"][count] = lm_train[:, 1].clone().detach().numpy()
        self.probs["Generative_test"][count] = lm_test[:, 1].clone().detach().numpy()
        self.probs["bucket_labels_train"][count] = self.probs["Generative_train"][count][self.unique_idx]
        self.mu_dict[count] = self.label_model.mu.clone().detach().numpy().squeeze()

        if self.final_model and count % self.discr_model_frequency == 0:
            self.metrics["Discriminative_train"][count] = self.final_model.analyze(self.y_train, fm_train)
            self.metrics["Discriminative_test"][count] = self.final_model.analyze(self.y_test, fm_test)
            self.probs["Discriminative_train"][count] = fm_train[:, 1].clone().cpu().detach().numpy()
            self.probs["Discriminative_test"][count] = fm_test[:, 1].clone().cpu().detach().numpy()

        if selected_point:
            self.queried.append(selected_point)
            if self.query_strategy in ["maxkl", "margin"] and self.randomness == 0:
                self.bucket_AL_values[count] = self.bucket_values

        return self
