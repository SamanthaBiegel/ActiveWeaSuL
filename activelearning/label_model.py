import numpy as np
import itertools
import torch
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from typing import Optional

from performance import PerformanceMixin


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
                 active_learning: bool = False,
                 penalty_strength: float = 1e3,
                 hide_progress_bar: bool = False):

        self.n_epochs = n_epochs
        self.lr = lr
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

        mask = np.ones((max(max(self.wl_idx.values()))+1, max(max(self.wl_idx.values()))+1))

        for key in self.wl_idx.keys():
            # Mask diagonal blocks
            mask[self.wl_idx[key][0]: self.wl_idx[key][-1] + 1, self.wl_idx[key][0]: self.wl_idx[key][-1] + 1] = 0

            key = key.split("_")

            # Create all possible subsets of clique
            clique_list = list(itertools.chain.from_iterable(
                itertools.combinations(key, r) for r in range(len(key) + 1) if r > 0))

            # Create all pairs of subsets of clique
            clique_pairs = list(itertools.permutations(["_".join(clique) for clique in clique_list], r=2))

            # Mask all pairs of subsets that are in the same clique
            for pair in clique_pairs:
                i = self.wl_idx[pair[0]]
                j = self.wl_idx[pair[1]]
                mask[i[0]:i[-1]+1, j[0]:j[-1]+1] = 0

        return mask

    def get_psi(self, label_matrix=None, cliques=None, nr_wl=None):
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

        psi_list = []
        col_counter = 0
        wl_idx = {}

        # Compute psi for individual weak labels
        for i in range(nr_wl):
            wl = label_matrix[:, i]
            wl_onehot = (wl[:, None] == self.y_set)*1
            psi_list.append(wl_onehot)
            wl_idx[str(i)] = list(range(col_counter, col_counter+wl_onehot.shape[1]))
            col_counter += wl_onehot.shape[1]

        psi = np.hstack(psi_list)

        # Compute psi for cliques
        psi_int_list = []
        clique_idx = {}
        # Iterate over maximal cliques
        for clique in cliques:
            # Compute set of all subcliques with at least 2 variables in maximal clique
            clique_comb = itertools.chain.from_iterable(
                itertools.combinations(clique, r) for r in range(len(clique)+1) if r > 1)
            for i, comb in enumerate(clique_comb):
                # Compute psi for clique of 2 variables
                if len(comb) == 2:
                    idx1 = wl_idx[str(comb[0])]
                    idx2 = wl_idx[str(comb[1])]
                    wl_int_onehot = (
                        (psi[:, None, idx1[0]:(idx1[-1]+1)]
                            * psi[:, idx2[0]:(idx2[-1]+1), None]).reshape(len(psi), -1)
                    )

                    psi_int_list.append(wl_int_onehot)
                    clique_idx[comb] = i
                    wl_idx[str(comb[0]) + "_" + str(comb[1])] = list(range(col_counter, col_counter+wl_int_onehot.shape[1]))
                
                # Compute psi for clique of 3 variables
                if len(comb) == 3:
                    idx3 = wl_idx[str(comb[2])]
                    wl_int_onehot = (
                        (psi_int_list[clique_idx[(comb[0], comb[1])]][:, None, :]
                            * psi[:, idx3[0]:(idx3[-1]+1), None]).reshape(len(psi), -1)
                    )
                    psi_int_list.append(wl_int_onehot)
                    wl_idx[str(comb[0]) + "_" + str(comb[1]) + "_" + str(comb[2])] = list(
                        range(col_counter, col_counter+wl_int_onehot.shape[1]))

                col_counter += wl_int_onehot.shape[1]

        # Concatenate different clique sizes
        if psi_int_list:
            psi_2 = np.hstack(psi_int_list)
            psi = np.concatenate([psi, psi_2], axis=1)

        return psi, wl_idx
        
    def init_label_model(self, label_matrix, cliques, class_balance):
        """Initialize label model"""

        self.label_matrix = label_matrix
        self.cliques = cliques
        self.class_balance = class_balance

        self.N, self.nr_wl = label_matrix.shape
        self.y_set = np.unique(label_matrix)  # array of classes

        # Ignore abstain label
        if -1 in self.y_set:
            self.y_set = self.y_set[self.y_set != -1]

        self.y_dim = len(self.y_set)  # number of classes

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
            _, self.bucket_idx, self.bucket_inverse, self.bucket_counts = np.unique(label_matrix, axis=0, return_index=True, return_inverse=True, return_counts=True)

        if not self.active_learning:
            self.z = nn.Parameter(torch.normal(0, 1, size=(self.psi.shape[1], self.y_dim - 1)), requires_grad=True)

        optimizer = torch.optim.Adam({self.z}, lr=self.lr)

        self.losses = []
        # Find z with SGD
        for epoch in tqdm(range(self.n_epochs), disable=self.hide_progress_bar):
            optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.clone().detach().numpy())

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
        psi, _ = self.get_psi(label_matrix=label_matrix, cliques=self.cliques, nr_wl=self.nr_wl)

        cliques_joined = self.cliques.copy()
        for i, clique in enumerate(cliques_joined):
            cliques_joined[i] = ["_".join(str(wl) for wl in clique)]
        self.max_clique_idx = np.array([idx for clique in cliques_joined for i, idx in enumerate(self.wl_idx[clique[0]])])
        clique_sums = torch.zeros((N, len(self.cliques)))
        for i, clique in enumerate(self.cliques):
            clique_sums[:, i] = torch.Tensor(label_matrix[:, clique] != -1).sum(dim=1) > 0
        n_cliques = clique_sums.sum(dim=1)

        # Product of weak label or clique probabilities per data point
        # Junction tree theorem
        psi_idx = torch.Tensor(psi[:, self.max_clique_idx].T)
        clique_probs = mu[self.max_clique_idx, :] * psi_idx
        clique_probs[psi_idx == 0] = 1
        P_joint_lambda_Y = torch.prod(clique_probs, dim=0)/(torch.tensor(P_Y) ** (n_cliques - 1))

        # Mask out data points with abstains in all cliques
        P_joint_lambda_Y[(clique_probs == 1).all(axis=0)] = np.nan

        # Marginal weak label probabilities
        lambda_combs, lambda_index, lambda_counts = np.unique(label_matrix, axis=0, return_counts=True, return_inverse=True)
        new_counts = lambda_counts.copy()
        rows_not_abstain, cols_not_abstain = np.where(lambda_combs != -1)
        for i, comb in enumerate(lambda_combs):
            nr_non_abstain = (comb != -1).sum()
            if nr_non_abstain < self.nr_wl:
                if nr_non_abstain == 0:
                    new_counts[i] = 0
                else:
                    match_rows = np.where((lambda_combs[:, cols_not_abstain[rows_not_abstain == i]] == lambda_combs[i, cols_not_abstain[rows_not_abstain == i]]).all(axis=1))       
                    new_counts[i] = lambda_counts[match_rows].sum()

        self.P_lambda = torch.Tensor((new_counts/N)[lambda_index][:, None])

        # Conditional label probability
        P_Y_given_lambda = (P_joint_lambda_Y[:, None] / self.P_lambda).clamp(0,1)

        prob_labels = torch.cat([1 - P_Y_given_lambda, P_Y_given_lambda], axis=1)

        if assign_train_labels:
            self.preds = prob_labels

        return prob_labels

    def predict_true(self, y_true, y_test=None, label_matrix=None):
        """Obtain training labels from optimal label model using ground truth labels"""

        if any(v is None for v in (label_matrix, y_test)):
            label_matrix = self.label_matrix
            y_test = y_true

        return self.predict(label_matrix, self.get_true_mu(y_true)[:, 1][:, None], y_test.mean())

    def predict_true_counts(self, y_true):
        """Obtain optimal training labels using ground truth labels"""

        lambda_combs, lambda_index, lambda_counts = np.unique(np.concatenate([self.label_matrix, y_true[:, None]], axis=1), axis=0, return_counts=True, return_inverse=True)

        P_Y_lambda = np.zeros((self.N, 2))

        for i, j in zip([0, 1], [1, 0]):
            P_Y_lambda[y_true == i, i] = ((lambda_counts/self.N)[lambda_index]/self.P_lambda.squeeze())[y_true == i]
            P_Y_lambda[y_true == i, j] = 1 - P_Y_lambda[y_true == i, i]

        return torch.Tensor(P_Y_lambda)

    def get_true_mu(self, y_true):
        """Obtain optimal label model parameters from data and ground truth labels"""

        exp_mu = np.zeros((self.psi.shape[1], self.y_dim))
        for i in range(0, self.y_dim):
            mean = self.psi[y_true == i].sum(axis=0) / self.N
            exp_mu[:, i] = mean

        return torch.Tensor(exp_mu)

    def get_true_cov_OS(self, y_true):
        """Obtain true covariance between cliques and Y using ground truth labels"""

        y_onehot = ((y_true[..., None] == self.y_set)*1).reshape((self.N, self.y_dim))
        psi_y = np.concatenate([self.psi, y_onehot], axis=1)

        cov_O_S = np.cov(psi_y.T, bias=True)

        return cov_O_S[:-self.y_dim, -self.y_dim:]