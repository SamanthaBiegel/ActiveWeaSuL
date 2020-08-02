import numpy as np
import itertools
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook as tqdm
from typing import Optional

from performance import ModelPerformance


class LabelModel(ModelPerformance):
    def __init__(self,
                 final_model_kwargs: dict,
                 df: pd.DataFrame,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 active_learning: bool = False,
                 add_cliques: bool = False,
                 add_prob_loss: bool = False):

        self.n_epochs = n_epochs
        self.lr = lr
        self.active_learning = active_learning
        self.add_cliques = add_cliques
        self.add_prob_loss = add_prob_loss
        self.final_model_kwargs = final_model_kwargs
        self.z = None
        super().__init__(df=df)
 
    def init_properties(self):
        """Get properties such as dimensions from label matrix"""

        self.N, self.nr_wl = self.label_matrix.shape
        self.y_set = np.unique(self.label_matrix)  # array of classes

        # Ignore abstain label
        if self.y_set[0] == -1:
            self.y_set = self.y_set[1:]

        self.y_dim = len(self.y_set)  # number of classes

    def calculate_mu(self, cov_OS):
        """Compute mu from OS covariance"""

        return cov_OS + torch.Tensor(self.E_O[:, np.newaxis] @ self.E_S[np.newaxis, ][:, np.newaxis])

    def calculate_cov_OS(self):
        """Compute unobserved part of covariance"""

        c = 1 / self.cov_S * (1 + torch.mm(torch.mm(self.z.T, self.cov_O), self.z))
        cov_OS = torch.mm(self.cov_O, self.z / torch.sqrt(c))

        return cov_OS

    def loss_prior_knowledge_cov(self, cov_OS, penalty_strength: float = 3):
        """Compute loss from prior knowledge on part of covariance matrix"""

        cov_OS_al = cov_OS[self.al_idx, :]

        return penalty_strength * torch.norm(cov_OS_al - self.cov_AL) ** 2

    def loss_prior_knowledge_probs(self, probs, penalty_strength: float = 1000):

        probs_al = ((self.y_set == self.ground_truth_labels[..., None]) * 1).reshape(self.N, -1)
        mask = (self.ground_truth_labels != -1)

        return penalty_strength * torch.norm(torch.Tensor((probs_al - probs))[mask, :]) ** 2
    
    def loss_probs(self, probs, penalty_strength: float = 1e3):

        loss_0 = torch.norm(torch.Tensor(probs[probs < 0])) ** 2
        loss_1 = torch.norm(torch.Tensor(probs[probs > 1] - 1)) ** 2

        return penalty_strength * (loss_0 + loss_1)

    def loss_func(self):
        """Compute loss for matrix completion problem"""

        loss = torch.norm((self.cov_O_inverse + self.z @ self.z.T)[torch.BoolTensor(self.mask)]) ** 2

        if self.add_prob_loss:
            tmp_cov = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov)
            tmp_probs = self._predict(tmp_mu)
            loss += self.loss_probs(tmp_probs)

        if self.active_learning == "cov":
            # Add loss for current covariance if taking active learning weak label into account
            tmp_cov_OS = self.calculate_cov_OS()
            loss += self.loss_prior_knowledge_cov(tmp_cov_OS)

        if self.active_learning == "probs":
            tmp_cov = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov)
            tmp_probs = self._predict(tmp_mu)
            loss += self.loss_prior_knowledge_probs(tmp_probs)

        return loss

    def create_mask(self):
        """Create mask to encode graph structure in covariance matrix"""

        if not self.add_cliques:
            mask = np.ones((self.nr_wl * self.y_dim, self.nr_wl * self.y_dim))
            for i in range(self.nr_wl):
                # Mask out diagonal blocks for the individual weak labels
                mask[i * self.y_dim:(i + 1) * self.y_dim, i * self.y_dim:(i + 1) * self.y_dim] = 0

            # Mask out interactions within cliques
            for clique in self.cliques:
                for pair in itertools.permutations(clique, r=2):
                    i = pair[0]
                    j = pair[1]
                    mask[i * self.y_dim:(i + 1) * self.y_dim, j * self.y_dim:(j + 1) * self.y_dim] = 0

            return mask

        else:
            mask = np.ones((max(max(self.wl_idx.values()))+1, max(max(self.wl_idx.values()))+1))

            for key in self.wl_idx.keys():
                mask[self.wl_idx[key][0]: self.wl_idx[key][-1] + 1, self.wl_idx[key][0]: self.wl_idx[key][-1] + 1] = 0

                key = key.split("_")

                # Create all possible subsets of clique
                clique_list = list(itertools.chain.from_iterable(
                    itertools.combinations(key, r) for r in range(len(key) + 1) if r > 0))

                # Create all pairs of subsets of clique
                clique_pairs = list(itertools.permutations(["_".join(clique) for clique in clique_list], r=2))

                for pair in clique_pairs:
                    i = self.wl_idx[pair[0]]
                    j = self.wl_idx[pair[1]]
                    mask[i[0]:i[-1]+1, j[0]:j[-1]+1] = 0

            return mask

    def get_psi(self):
        """Transform label matrix to indicator variables"""

        psi_list = []
        col_counter = 0
        wl_idx = {}
        for i in range(self.nr_wl):
            wl = self.label_matrix[:, i]
            wl_onehot = (wl[:, np.newaxis] == self.y_set)*1
            psi_list.append(wl_onehot)
            wl_idx[str(i)] = list(range(col_counter, col_counter+wl_onehot.shape[1]))
            col_counter += wl_onehot.shape[1]

        psi = np.hstack(psi_list)

        if not self.add_cliques:
            return psi, wl_idx

        psi_int_list = []
        clique_idx = {}
        for clique in self.cliques:
            clique_comb = itertools.chain.from_iterable(
                itertools.combinations(clique, r) for r in range(len(clique)+1) if r > 1)
            for i, comb in enumerate(clique_comb):
                if len(comb) == 2:
                    idx1 = wl_idx[str(comb[0])]
                    idx2 = wl_idx[str(comb[1])]
                    wl_int_onehot = (
                        (psi[:, np.newaxis, idx1[0]:(idx1[-1]+1)]
                            * psi[:, idx2[0]:(idx2[-1]+1), np.newaxis]).reshape(len(psi), -1)
                    )

                    psi_int_list.append(wl_int_onehot)
                    clique_idx[comb] = i
                    wl_idx[str(comb[0]) + "_" + str(comb[1])] = list(range(col_counter, col_counter+wl_int_onehot.shape[1]))

                if len(comb) == 3:
                    idx3 = wl_idx[str(comb[2])]
                    wl_int_onehot = (
                        (psi_int_list[clique_idx[(comb[0], comb[1])]][:, np.newaxis, :]
                            * psi[:, idx3[0]:(idx3[-1]+1), np.newaxis]).reshape(len(psi), -1)
                    )
                    psi_int_list.append(wl_int_onehot)
                    wl_idx[str(comb[0]) + "_" + str(comb[1]) + "_" + str(comb[2])] = list(
                        range(col_counter, col_counter+wl_int_onehot.shape[1]))

                col_counter += wl_int_onehot.shape[1]

        psi_2 = np.hstack(psi_int_list)

        return np.concatenate([psi, psi_2], axis=1), wl_idx

    def init_cov_exp(self):
        "Compute expectations and covariances for observed set and separator set"

        # Compute observed expectations and covariances
        self.E_O = self.psi.mean(axis=0)
        self.cov_O = torch.Tensor(np.cov(self.psi.T, bias=True))
        self.cov_O_inverse = torch.Tensor(np.linalg.pinv(self.cov_O.numpy()))
        # self.cov_O_inverse = torch.pinverse(self.cov_O)

        self.E_S = np.array(self.class_balance[-1])
        self.cov_Y = np.diag(self.class_balance) - self.class_balance.reshape((-1, 1)) @ self.class_balance.reshape((1, -1))
        # In the rank-one setting we only consider one column of psi(Y)
        self.cov_S = self.cov_Y[-1, -1]

        # Marginal weak label probabilities
        _, lambda_index, lambda_counts = np.unique(self.label_matrix, axis=0, return_counts=True, return_inverse=True)
        P_lambda = lambda_counts/self.N
        self.P_lambda = torch.Tensor(P_lambda[lambda_index][:, np.newaxis])

    def fit(self,
            label_matrix,
            cliques,
            class_balance,
            ground_truth_labels: Optional[np.array] = None):
        """Fit label model"""

        writer = SummaryWriter()
        
        self.label_matrix = label_matrix
        self.cliques = cliques
        self.class_balance = class_balance

        self.init_properties()

        # Transform label matrix to indicator variables
        self.psi, self.wl_idx = self.get_psi()
        self.init_cov_exp()

        self.mask = self.create_mask()

        if self.active_learning is not False:
            # Calculate known covariance for active learning weak label
            self.al_idx = self.wl_idx[str(self.nr_wl-1)]
            self.cov_AL = torch.Tensor((self.psi[:, self.al_idx] * self.psi[:, self.al_idx]).mean(axis=0) / self.class_balance.reshape(-1, 1) * self.cov_Y)
            self.ground_truth_labels = ground_truth_labels

        if self.z is None:
            self.z = nn.Parameter(torch.normal(0, 1, size=(self.psi.shape[1], self.y_dim - 1)), requires_grad=True)

        optimizer = torch.optim.Adam({self.z}, lr=self.lr)

        # Find optimal z
        for epoch in tqdm(range(self.n_epochs)):
            optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            optimizer.step()

            writer.add_scalar('label model loss', loss, epoch)

            tmp_cov_OS = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov_OS)
            tmp_probs = self._predict(tmp_mu)
            writer.add_scalar('label model accuracy', self._accuracy(tmp_probs, self.df["y"].values), epoch)

            # if epoch == 0 or epoch % 25 == 24:
            #     final_probs = fit_predict_fm(self.df[["x1", "x2"]].values, tmp_probs, **self.final_model_kwargs, soft_labels=True)
            #     writer.add_scalar('final model accuracy', self._accuracy(final_probs, self.df["y"].values), epoch)

        # Compute covariances and label model probabilities from optimal z
        self.cov_OS = self.calculate_cov_OS()
        self.mu = self.calculate_mu(self.cov_OS)

        writer.flush()
        writer.close()

        return self

    def predict(self):
        """Predict training labels"""

        return self._predict(self.mu)

    def _predict(self, mu):

        if not self.add_cliques:
            idx = np.array(range(self.nr_wl*self.y_dim))
        else:
            cliques_joined = self.cliques.copy()
            for i, clique in enumerate(cliques_joined):
                cliques_joined[i] = ["_".join(str(wl) for wl in clique)]
            idx = np.array([idx for clique in cliques_joined for i, idx in enumerate(self.wl_idx[clique[0]])])

        # Product of weak label or clique probabilities per data point
        # Junction tree theorem
        # P_joint_lambda_Y = (np.prod(np.tile(mu[idx, :].T, (self.N, 1)), axis=1, where=(self.psi[:, idx] == 1))
        #                     / self.E_S)

        clique_probs = mu[idx, :] * torch.Tensor(self.psi[:, idx].T)
        clique_probs[clique_probs == 0] = 1
        P_joint_lambda_Y = torch.prod(clique_probs, dim=0)/torch.Tensor(self.E_S)

        # Conditional label probability
        P_Y_given_lambda = (P_joint_lambda_Y[:, None] / self.P_lambda)

        preds = torch.cat([1 - P_Y_given_lambda, P_Y_given_lambda], axis=1)

        if self._accuracy(preds, self.df["y"].values) < 0.5:
            preds[:, [1, 0]] = preds[:, [0, 1]]

        return preds

    def get_true_mu(self):
        """Obtain actual label model parameters from data and ground truth labels"""

        exp_mu = np.zeros((self.psi.shape[1], self.y_dim))
        for i in range(self.y_dim):
            mean = self.psi[self.df["y"].values == i].sum(axis=0) / self.N
            exp_mu[:, i] = mean

        return exp_mu


# def get_conditional_probabilities(label_matrix, mu):
#     """Get conditional probabilities from label model parameters"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     c_probs = np.zeros(((nr_wl) * (y_dim + 1), y_dim))
#     for wl in range(nr_wl):
#         # Conditional probabilities are label model parameters
#         c_probs[(wl * y_dim) + wl + 1:y_dim + (wl * y_dim) + wl + 1, :] = mu[(wl * y_dim):y_dim + (wl * y_dim), :]

#         # Probability for abstain
#         c_probs[(wl * y_dim) + wl, :] = 1 - mu[(wl * y_dim):y_dim + (wl * y_dim), :].sum(axis=0)

#     return c_probs


# def get_accuracies(label_matrix, c_probs, class_balance):
#     """Get weak label accuracies"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     # Joint probabilities from conditional
#     E_S = (y_set * class_balance).sum()
#     P_Ylam = c_probs * np.tile(E_S, ((y_dim + 1) * nr_wl, 1))

#     weights = np.zeros((1, nr_wl))
#     for i in range(nr_wl):
#         # Sum of probabilities where weak label and Y agree
#         weights[:, i] = P_Ylam[i * (y_dim + 1) + 1, 0] + P_Ylam[i * (y_dim + 1) + 2, 1]

#     # Label propensity of weak labels
#     coverage = (label_matrix != -1).mean(axis=0)

#     return np.clip(weights / coverage, 1e-6, 1)


# def get_true_accuracies(label_matrix, y):
#     """Obtain actual weak label accuracies from data and ground truth labels"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     coverage = (label_matrix != -1).mean(axis=0)

#     true_accuracies = np.zeros((1, nr_wl))
#     for i in range(nr_wl):
#         true_accuracies[:, i] = (label_matrix[:, i] == y).mean()

#     return true_accuracies/coverage