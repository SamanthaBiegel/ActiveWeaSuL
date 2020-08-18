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
                 add_prob_loss: bool = False,
                 hide_progress_bar: bool = False):

        self.n_epochs = n_epochs
        self.lr = lr
        self.active_learning = active_learning
        self.add_cliques = add_cliques
        self.add_prob_loss = add_prob_loss
        self.hide_progress_bar = hide_progress_bar
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

        return cov_OS + torch.Tensor(self.E_O[:, np.newaxis] @ self.E_S[np.newaxis, np.newaxis])

    def calculate_cov_OS(self):
        """Compute unobserved part of covariance"""

        # cov_S^-1, cov_S is a scalar
        c = 1 / self.cov_S * (1 + torch.mm(torch.mm(self.z.T, self.cov_O), self.z))
        cov_OS = torch.mm(self.cov_O, self.z / torch.sqrt(c))

        return cov_OS

    def loss_prior_knowledge_cov(self, cov_OS, penalty_strength: float = 1e4):
        """Compute loss from prior knowledge on part of covariance matrix"""

        # cov_OS_al = cov_OS[list(itertools.chain.from_iterable([self.wl_idx[clique] for clique in ["3", "0_3", "1_3", "2_3", "1_2_3"]]))]
        cov_OS_al = cov_OS[self.al_idx]

        return penalty_strength * torch.norm(cov_OS_al - self.cov_AL[:, np.newaxis]) ** 2

    def loss_prior_knowledge_probs(self, probs, penalty_strength: float = 1e3):

        probs_al = torch.Tensor(((self.y_set == self.ground_truth_labels[..., None]) * 1).reshape(self.N, -1))
        mask = (self.ground_truth_labels != -1)

        return penalty_strength * torch.norm((probs_al - probs)[mask, :]) ** 2
    
    def loss_probs(self, probs, penalty_strength: float = 1e6):

        loss_0 = torch.norm(torch.Tensor(probs[probs < 0]))
        loss_1 = torch.norm(torch.Tensor(probs[probs > 1] - 1))
        print(loss_0+loss_1)
        return penalty_strength * (loss_0 + loss_1)

    def loss_func(self):
        """Compute loss for matrix completion problem"""

        loss = torch.norm((self.cov_O_inverse + self.z @ self.z.T)[torch.BoolTensor(self.mask)]) ** 2

        if self.add_prob_loss:
            tmp_cov = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov)
            tmp_probs = self._predict(tmp_mu, torch.tensor(self.E_S))
            loss += self.loss_probs(tmp_probs)

        if self.active_learning == "cov":
            # Add loss for current covariance if taking active learning weak label into account
            tmp_cov_OS = self.calculate_cov_OS()
            loss += self.loss_prior_knowledge_cov(tmp_cov_OS)

        if self.active_learning == "probs":
            tmp_cov = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov)
            tmp_probs = self._predict(tmp_mu, torch.tensor(self.E_S))
            loss += self.loss_prior_knowledge_probs(tmp_probs)

        return loss

    def create_mask(self):
        """Create mask to encode graph structure in covariance matrix"""

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

        if not self.add_cliques:
            # Mask out interactions within cliques
            for clique in self.cliques:
                for pair in itertools.permutations(clique, r=2):
                    i = pair[0]
                    j = pair[1]
                    mask[i * self.y_dim:(i + 1) * self.y_dim, j * self.y_dim:(j + 1) * self.y_dim] = 0

        return mask

    def get_psi(self):

        return self._get_psi(self.label_matrix)

    def _get_psi(self, label_matrix):
        """Transform label matrix to indicator variables"""

        psi_list = []
        col_counter = 0
        wl_idx = {}
        for i in range(self.nr_wl):
            wl = label_matrix[:, i]
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

        psi = np.concatenate([psi, psi_2], axis=1)

        # wl_idx = {k: v for k, v in wl_idx.items() if k in ["0", "1", "2", "3", "1_2"]}

        # psi = np.concatenate([psi, psi_2], axis=1)[:, list(itertools.chain.from_iterable([wl_idx[clique] for clique in wl_idx.keys()]))]

        # wl_idx["1_2"] = [8,9,10,11]

        return psi, wl_idx

    def init_cov_exp(self):
        "Compute expectations and covariances for observed set and separator set"

        # Compute observed expectations and covariances
        self.E_O = self.psi.mean(axis=0)
        cov_O = np.cov(self.psi.T, bias=True)
        self.cov_O_inverse = torch.Tensor(np.linalg.pinv(cov_O))
        self.cov_O = torch.Tensor(cov_O)
        # self.cov_O_inverse = torch.pinverse(self.cov_O)

        self.E_S = np.array(self.class_balance[-1])
        self.cov_Y = np.diag(self.class_balance) - self.class_balance.reshape((-1, 1)) @ self.class_balance.reshape((1, -1))
        # In the rank-one setting we only consider one column of psi(Y)
        self.cov_S = self.cov_Y[-1, -1]

        # Marginal weak label probabilities
        if self.active_learning == "cov" and self.add_cliques:
            _, lambda_index, lambda_counts = np.unique(self.label_matrix[:, :3], axis=0, return_counts=True, return_inverse=True)
        else:
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
            self.ground_truth_labels = ground_truth_labels

            if self.active_learning == "cov":
                # self.cov_AL = torch.Tensor((self.psi[:, self.al_idx] * self.psi[:, self.al_idx]).mean(axis=0) / self.class_balance.reshape(-1, 1) * self.cov_Y)[:, 1]
                E_AL_Y = self.E_O.copy()
                # E_AL_Y = self.psi[:, self.al_idx].mean(axis=0)
                E_AL_Y[self.al_idx[0]] = 0
                self.cov_AL = torch.Tensor(E_AL_Y[self.al_idx] - self.psi[:, self.al_idx].mean(axis=0)*self.E_S)

                # E_AL_Y[self.wl_idx["0_3"][0:2]] = 0
                # self.cov_AL_03 = torch.Tensor(E_AL_Y[self.wl_idx["0_3"]] - self.psi[:, self.wl_idx["0_3"]].mean(axis=0)*self.E_S)

                # E_AL_Y[self.wl_idx["1_3"][0:2]] = 0
                # self.cov_AL_13 = torch.Tensor(E_AL_Y[self.wl_idx["1_3"]] - self.psi[:, self.wl_idx["1_3"]].mean(axis=0)*self.E_S)

                # E_AL_Y[self.wl_idx["2_3"][0:2]] = 0
                # self.cov_AL_23 = torch.Tensor(E_AL_Y[self.wl_idx["2_3"]] - self.psi[:, self.wl_idx["2_3"]].mean(axis=0)*self.E_S)

                # E_AL_Y[self.wl_idx["1_2_3"][0:4]] = 0
                # self.cov_AL_123 = torch.Tensor(E_AL_Y[self.wl_idx["1_2_3"]] - self.psi[:, self.wl_idx["1_2_3"]].mean(axis=0)*self.E_S)

                # self.cov_AL = torch.cat((self.cov_AL_3, self.cov_AL_03, self.cov_AL_13, self.cov_AL_23, self.cov_AL_123))
                # self.cov_AL = torch.cat((self.cov_AL_3, self.cov_AL_03, self.cov_AL_13, self.cov_AL_23))

        if self.z is None:
            self.z = nn.Parameter(torch.normal(0, 1, size=(self.psi.shape[1], self.y_dim - 1)), requires_grad=True)

        if self.cov_O_inverse.shape[0] != self.z.shape[0]:
            self.z = nn.Parameter(torch.cat((self.z, torch.normal(0, 1, size=(self.y_dim, self.y_dim - 1)))), requires_grad=True)

        optimizer = torch.optim.Adam({self.z}, lr=self.lr)
        # lambda1 = lambda epoch: 0.999 ** epoch
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        # Find optimal z
        for epoch in tqdm(range(self.n_epochs), disable=self.hide_progress_bar):
            optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            writer.add_scalar('label model loss', loss, epoch)

            tmp_cov_OS = self.calculate_cov_OS()
            tmp_mu = self.calculate_mu(tmp_cov_OS)
            tmp_probs = self._predict(tmp_mu, torch.tensor(self.E_S))
            writer.add_scalar('label model accuracy', self._accuracy(tmp_probs, self.df["y"].values), epoch)

            # if epoch == 0 or epoch % 25 == 24:
            #     final_probs = fit_predict_fm(self.df[["x1", "x2"]].values, tmp_probs, **self.final_model_kwargs, soft_labels=True)
            #     writer.add_scalar('final model accuracy', self._accuracy(final_probs, self.df["y"].values), epoch)

        # Determine the sign of z
        # Assuming cov_OS corresponds to Y=1, then cov(wl1=0,Y=1) should be negative
        # If not, flip signs to get the covariance for Y=1
        if self.calculate_cov_OS()[0] > 0:
            self.z = -self.z

        # Compute covariances and label model probabilities from optimal z
        self.cov_OS = self.calculate_cov_OS()
        self.mu = self.calculate_mu(self.cov_OS)#.clamp(0, 1)

        writer.flush()
        writer.close()

        return self

    def predict(self):
        """Predict training labels"""

        return self._predict(self.mu, torch.tensor(self.E_S))

    def _predict(self, mu, P_Y):

        if not self.add_cliques:
            idx = np.array(range(self.nr_wl*self.y_dim))
            n_cliques = self.nr_wl
        elif self.active_learning == "cov" and self.add_cliques:
            idx = list(itertools.chain.from_iterable([self.wl_idx[clique] for clique in ["0", "1_2"]]))
            n_cliques = 2
        else:
            cliques_joined = self.cliques.copy()
            for i, clique in enumerate(cliques_joined):
                cliques_joined[i] = ["_".join(str(wl) for wl in clique)]
            idx = np.array([idx for clique in cliques_joined for i, idx in enumerate(self.wl_idx[clique[0]])])
            n_cliques = len(cliques_joined)

        # Product of weak label or clique probabilities per data point
        # Junction tree theorem
        clique_probs = mu[idx, :] * torch.Tensor(self.psi[:, idx].T)
        clique_probs[clique_probs == 0] = 1
        P_joint_lambda_Y = torch.prod(clique_probs, dim=0)/(P_Y ** (n_cliques - 1))

        # Conditional label probability
        P_Y_given_lambda = (P_joint_lambda_Y[:, None] / self.P_lambda)#.clamp(0,1)

        preds = torch.cat([1 - P_Y_given_lambda, P_Y_given_lambda], axis=1)

        return preds

    def predict_true(self):
        
        return self._predict(self.get_true_mu()[:, 1][:, np.newaxis], self.df["y"].mean())

    def get_true_mu(self):
        """Obtain actual label model parameters from data and ground truth labels"""

        exp_mu = np.zeros((self.psi.shape[1], self.y_dim))
        for i in range(0, self.y_dim):
            mean = self.psi[self.df["y"].values == i].sum(axis=0) / self.N
            exp_mu[:, i] = mean

        return torch.Tensor(exp_mu)

    def get_true_cov_OS(self):

        y_onehot = ((self.df["y"].values[..., None] == self.y_set)*1).reshape((self.N, self.y_dim))
        psi_y = np.concatenate([self.psi, y_onehot], axis=1)

        cov_O_S = np.cov(psi_y.T, bias=True)

        return cov_O_S[:-self.y_dim, -self.y_dim:]




        



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