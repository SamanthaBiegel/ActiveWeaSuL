import numpy as np
import itertools
import torch
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from typing import Optional

from itertools import product, chain
import pandas as pd

from activeweasul.performance import PerformanceMixin


class LabelModel(PerformanceMixin):
    """Fit label model using Matrix Completion approach (Ratner et al. 2019).
    Optionally, add penalty for labeled points (active learning).

    Args:
        n_epochs (int, optional): Number of epochs
        lr (float, optional): Learning rate
        penalty_strength (float, optional): Strength of the active learning penalty
        active_learning (bool, optional): Add active learning component
        hide_progress_bar (bool, optional): Hide epoch progress bar
    """

    def __init__(self,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 cardinality: int = 2,
                 active_learning: bool = False,
                 penalty_strength: float = 1e3,
                 hide_progress_bar: bool = False):

        self.n_epochs = n_epochs
        self.lr = lr
        self.cardinality = cardinality
        self.active_learning = active_learning
        self.penalty_strength = penalty_strength
        self.hide_progress_bar = hide_progress_bar

    def calculate_mu(self, cov_OS):
        """Compute mu from OS covariance"""

        return cov_OS + torch.Tensor(self.E_O[:, None] @ self.E_S[None, None])

    def calculate_cov_OS(self):
        """Compute unobserved part of covariance"""

        # cov_S^-1, cov_S is a scalar
        c = 1 / self.cov_S * (1 + torch.mm(torch.mm(self.z.T, self.cov_O), self.z))
        cov_OS = torch.mm(self.cov_O, self.z / torch.sqrt(c))

        return cov_OS

    def loss_prior_knowledge_probs(self, probs):
        """Add penalty to loss for sampled data points"""

        # Label to probabilistic label (eg 1 to (0 1))
        # probs_al = torch.Tensor(((self.y_set == self.ground_truth_labels[..., None]) * 1).reshape(self.N, -1))

        # Select points for which ground truth label is available
        mask = (self.ground_truth_labels != -1)
        # nr_iterations = mask.sum()
        # discount_factor = torch.tensor(0.97).repeat(nr_iterations) ** torch.arange(nr_iterations-1, -1, -1)

        # loss = nn.CrossEntropyLoss(reduction="sum")
        # return penalty_strength * loss(probs[mask], torch.LongTensor(self.ground_truth_labels[mask]))

        # loss = nn.CrossEntropyLoss(reduction="none")
        # return penalty_strength * torch.sum(1/torch.Tensor(self.bucket_counts[self.bucket_inverse][mask]) * loss(probs[mask], torch.LongTensor(self.ground_truth_labels[mask])))

        # return penalty_strength * torch.sum(discount_factor * ((torch.Tensor(self.ground_truth_labels) - probs[:,1])[mask] ** 2))

        return self.penalty_strength * torch.norm((torch.Tensor(self.ground_truth_labels) - probs[:,1])[mask]) ** 2

    def loss_func(self):
        """Compute loss for matrix completion problem"""

        loss = torch.norm((self.cov_O_inverse + self.z @ self.z.T)[torch.BoolTensor(self.mask)]) ** 2

        if self.active_learning:
            tmp_cov = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov)
            tmp_probs = self.predict(self.label_matrix, tmp_mu, self.E_S)
            loss += self.loss_prior_knowledge_probs(tmp_probs)
            # if self.last_posteriors is not None:
            #     # print(torch.max((torch.norm(tmp_probs[self.bucket_idx, 1] - torch.Tensor(self.last_posteriors)) - 0.3), 0).values)
            #     loss += 1e3 * torch.max((torch.norm(tmp_probs[self.bucket_idx, 1] - torch.Tensor(self.last_posteriors)) - 0.4), 0).values

        return loss

    def create_mask(self):
        """Create mask to encode graph structure in covariance matrix"""

        mask = np.ones((self.psi.shape[1], self.psi.shape[1]))
        for idx in self.wl_idx.values():
            mask[np.ix_(idx, idx)] = 0
        return mask

    def get_clique_combinations(
            self,
            L,
            clique,
            # remove_all_zero=True,
            debug=False
        ):
        """

        Args:
            L ([type]): [description]
            clique ([type]): [description]
            remove_all_zero (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        choices = [-1] + list(range(self.cardinality))
        # Generate all combinations of outputs
        list_comb = product(*[choices] * len(clique))
        # Remove the combination where all elements are abstain (-1)
        comb_drop = tuple([-1]*len(clique))
        list_comb = [c for c in list_comb if c != comb_drop]
        # Generate the columns for each combination
        new_columns = []
        for comb in list_comb:
            new_columns.append(
                np.all(np.equal(L[:, clique], comb), axis=1)
            )
        # Combine into a matrix
        L_clique = np.vstack(new_columns).T

        if debug:
            return list_comb, L_clique.astype(int)

        return L_clique.astype(int)

    def get_psi(
        self,
        label_matrix=None,
        cliques=None,
        nr_wl=None,
        training=True
    ):
        """Compute psi from given label matrix and cliques

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on dataset
            cliques (list): List of lists of maximal cliques (column indices of label matrix)
            nr_wl (int): Number of weak labels

        Returns:
            numpy.array: Array of indicator variables
            dict: Mapping from clique to column index in psi
        """
        if any(v is None for v in (label_matrix, cliques, nr_wl)):
            label_matrix = self.label_matrix
            cliques = self.cliques
            nr_wl = self.nr_wl

        # Generate one array per maximal clique in `cliques`
        psi = [
            self.get_clique_combinations(label_matrix, clique)
            for clique in cliques
        ]

        # The indices of the cliques in psi are computed only during training
        if training:
            last_index = 0
            self.wl_idx = {}

            for clique, psi_cols in zip(cliques, psi):
                key = "_".join(str(c) for c in clique)
                # Index of the columns with at least one non-zero entry
                non_zero_idx = psi_cols.sum(axis=0) != 0
                n_non_zero_elements = sum(non_zero_idx)
                self.wl_idx[key] = list(
                    range(last_index, last_index + n_non_zero_elements)
                )
                last_index += n_non_zero_elements

        # Combine the arrays to get psi
        psi = np.hstack(psi)

        # Select only the columns with non-zero elements
        psi = psi[:, list(chain.from_iterable(self.wl_idx.values()))]
        # ! I kept the return statement as it was for compatibility reasons
        return psi, self.wl_idx

    def init_label_model(self, label_matrix, cliques, class_balance):
        """Initialize label model"""

        self.label_matrix = label_matrix
        self.cliques = cliques
        self.class_balance = class_balance

        self.N, self.nr_wl = label_matrix.shape
        self.y_set = np.unique(label_matrix)  # array of classes

        self.psi, self.wl_idx = self.get_psi()

        # Compute observed expectations and covariances
        self.E_O = self.psi.mean(axis=0)
        cov_O = np.cov(self.psi.T, bias=True)
        self.cov_O_inverse = torch.Tensor(np.linalg.pinv(cov_O))
        self.cov_O = torch.Tensor(cov_O)
        # self.cov_O_inverse = torch.pinverse(self.cov_O)

        self.E_S = np.array(self.class_balance[-1])
        self.cov_Y = np.diag(self.class_balance) - self.class_balance[:, None] @ self.class_balance[None, :]
        # In the rank-one setting we only consider one column of psi(Y)
        self.cov_S = self.cov_Y[-1, -1]

        self.mask = self.create_mask()

    def fit(self,
            label_matrix,
            cliques,
            class_balance,
            ground_truth_labels: Optional[np.array] = None):
            # last_posteriors: Optional[np.array] = None):
        """Fit label model

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on train set
            cliques (list): List of lists of maximal cliques (column indices of label matrix)
            class_balance (numpy.array): Array with true class distribution
            ground_truth_labels (numpy.array, optional): Array with -1 or ground truth label for each point in train set

        Returns:
            [type]: [description]
        """

        self.init_label_model(label_matrix, cliques, class_balance)

        if self.active_learning:
            self.ground_truth_labels = ground_truth_labels
            # self.last_posteriors = last_posteriors
            _, self.bucket_idx, self.bucket_inverse, self.bucket_counts =\
                np.unique(
                    label_matrix, axis=0, return_index=True,
                    return_inverse=True, return_counts=True
                )

        if not self.active_learning:
            self.z = nn.Parameter(
                torch.normal(
                    0, 1, size=(self.psi.shape[1], self.cardinality-1)
                ), requires_grad=True
            )

        optimizer = torch.optim.Adam({self.z}, lr=self.lr)

        self.losses = []
        # Find z with SGD
        for epoch in tqdm(range(self.n_epochs), disable=self.hide_progress_bar):
            optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.clone().detach().numpy())
        self.losses = np.array(self.losses)

        # Determine the sign of z
        # Assuming cov_OS corresponds to Y=1, then cov(wl1=1,Y=1) should be positive
        # If not, flip signs to get the covariance for Y=1
        if self.calculate_cov_OS()[1] < 0:
            self.z = nn.Parameter(-self.z, requires_grad=True)

        # Compute covariances and label model probabilities from z
        self.cov_OS = self.calculate_cov_OS()
        self.mu = self.calculate_mu(self.cov_OS)#.clamp(0, 1)

        return self

    def predict(self, label_matrix=None, mu=None, P_Y=None, assign_train_labels=False):
        """Predict probabilistic labels for a dataset from given parameters and class balance

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on dataset
            mu (torch.Tensor): Tensor with label model parameters
            P_Y (float): Estimated probability of Y=1 for dataset

        Returns:
            torch.Tensor: Tensor with probabilistic labels for given dataset
        """

        if any(v is None for v in (label_matrix, mu, P_Y)):
            label_matrix = self.label_matrix
            mu = self.mu
            P_Y = self.E_S
            assign_train_labels = True

        N = label_matrix.shape[0]
        psi, _ = self.get_psi(
            label_matrix=label_matrix, cliques=self.cliques, nr_wl=self.nr_wl,
            training=False
        )

        cliques_joined = self.cliques.copy()
        for i, clique in enumerate(cliques_joined):
            cliques_joined[i] = ["_".join(str(wl) for wl in clique)]
        self.max_clique_idx = np.array(
            [idx for clique in cliques_joined
             for i, idx in enumerate(self.wl_idx[clique[0]])]
        )
        clique_sums = torch.zeros((N, len(self.cliques)))
        for i, clique in enumerate(self.cliques):
            clique_sums[:, i] = torch.Tensor(label_matrix[:, clique] != -1).sum(dim=1) > 0
        n_cliques = clique_sums.sum(dim=1)

        # Product of weak label or clique probabilities per data point
        # Junction tree theorem
        psi_idx = torch.Tensor(psi[:, self.max_clique_idx].T)
        clique_probs = mu[self.max_clique_idx, :] * psi_idx
        clique_probs[psi_idx == 0] = 1
        P_joint_lambda_Y = (
            torch.prod(clique_probs, dim=0)/
            (torch.tensor(P_Y) ** (n_cliques - 1))
        )

#         # Mask out data points with abstains in all cliques
#         P_joint_lambda_Y[(clique_probs == 1).all(axis=0)] = np.nan

        # Marginal weak label probabilities
        lambda_combs, lambda_index, lambda_counts = np.unique(
            label_matrix, axis=0, return_counts=True, return_inverse=True
        )

        # new_counts = lambda_counts.copy()
        # rows_not_abstain, cols_not_abstain = np.where(lambda_combs != -1)
        # for i, comb in enumerate(lambda_combs):
        #     nr_non_abstain = (comb != -1).sum()
        #     if nr_non_abstain < self.nr_wl:
        #         if nr_non_abstain == 0:
        #             new_counts[i] = 0
        #         else:
        #             match_rows = np.where(
        #                 (lambda_combs[:, cols_not_abstain[rows_not_abstain == i]] == lambda_combs[i, cols_not_abstain[rows_not_abstain == i]]).all(axis=1))
        #             new_counts[i] = lambda_counts[match_rows].sum()

        # self.P_lambda = torch.Tensor((new_counts/N)[lambda_index][:, None])

        self.P_lambda = torch.Tensor((lambda_counts/N)[lambda_index][:, None])


        # Conditional label probability
        P_Y_given_lambda = (P_joint_lambda_Y[:, None] / self.P_lambda).clamp(0,1)

        prob_labels = torch.cat([1 - P_Y_given_lambda, P_Y_given_lambda], axis=1)

        if assign_train_labels:
            self.preds = prob_labels

        return prob_labels

    def predict_true(self, y_true):
        """Obtain training labels from optimal label model using ground truth labels"""

        return self.predict(self.label_matrix, self.get_true_mu(y_true)[:, 1][:, None], y_true.mean())

    def predict_true_counts(self, y_true):
        """Obtain optimal training labels using ground truth labels"""

        lambda_combs, lambda_index, lambda_counts = np.unique(np.concatenate([self.label_matrix[:,:3], y_true[:, None]], axis=1), axis=0, return_counts=True, return_inverse=True)

        P_Y_lambda = np.zeros((self.N, 2))

        for i, j in zip([0, 1], [1, 0]):
            P_Y_lambda[y_true == i, i] = ((lambda_counts/self.N)[lambda_index]/self.P_lambda.squeeze())[y_true == i]
            P_Y_lambda[y_true == i, j] = 1 - P_Y_lambda[y_true == i, i]

        return torch.Tensor(P_Y_lambda)

    def get_true_mu(self, y_true):
        """Obtain optimal label model parameters from data and ground truth labels"""

        exp_mu = np.zeros((self.psi.shape[1], self.cardinality))
        for i in range(0, self.cardinality):
            mean = self.psi[y_true == i].sum(axis=0) / self.N
            exp_mu[:, i] = mean

        return torch.Tensor(exp_mu)

    def get_true_cov_OS(self, y_true):
        """Obtain true covariance between cliques and Y using ground truth labels"""

        y_onehot = ((y_true[..., None] == self.y_set)*1).reshape((self.N, self.cardinality))
        psi_y = np.concatenate([self.psi, y_onehot], axis=1)

        cov_O_S = np.cov(psi_y.T, bias=True)

        return cov_O_S[:-self.cardinality, -self.cardinality:]