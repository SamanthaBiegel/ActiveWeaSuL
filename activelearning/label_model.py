import numpy as np
import itertools
import torch
import torch.nn as nn


def get_properties(label_matrix):
    """Get properties such as dimensions from label matrix"""

    N, nr_wl = label_matrix.shape
    y_set = np.unique(label_matrix)  # array of classes
    if y_set[0] == -1:
        y_set = y_set[1:]
    y_dim = len(y_set)  # number of classes

    return N, nr_wl, y_set, y_dim


def calculate_mu(cov_OS, exp_S, exp_O):
    """Compute mu from OS covariance"""

    return (cov_OS + torch.Tensor(exp_O.reshape(-1, 1) @ exp_S.reshape(1, -1))) / torch.Tensor(
        np.tile(exp_S, (cov_OS.shape[0], 1)))


def calculate_cov_OS(z, cov_S, cov_O):
    """Compute unobserved part of covariance"""

    c = 1 / cov_S * (1 + torch.mm(torch.mm(z.T, cov_O), z))
    cov_OS = torch.mm(cov_O, z / torch.sqrt(c))

    # Add covariance for opposite label
    if cov_OS[0] < 0:
        joint_cov_OS = torch.cat((-1 * cov_OS, cov_OS), axis=1)
    else:
        joint_cov_OS = torch.cat((cov_OS, -1 * cov_OS), axis=1)

    return joint_cov_OS


def loss_pk_cov(cov_OS, cov_AL, s: float = 3):
    """Compute loss from prior knowledge on part of covariance matrix"""

    O_dim, y_dim = cov_OS.shape

    cov_OS_known = torch.zeros(cov_OS.shape)
    cov_OS_known[-y_dim:, :] = cov_AL

    # Mask for active learning weak label part of covariance
    m = torch.ones([O_dim, y_dim], dtype=torch.bool)
    m[:-y_dim, :] = False

    return s * torch.norm((cov_OS - cov_OS_known)[m]) ** 2


def loss_pk_probs(probs, y_al, label_matrix, s: float = 0.5):

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    probs_al = ((y_set == y_al[..., None]) * 1).reshape(N_total, -1)
    mask = (y_al != -1)

    return s * torch.norm(torch.Tensor(probs_al - probs)[mask, :]) ** 2


def loss_func(z, label_matrix, cov_S, cov_O, cov_O_inverse, E_S, E_O, mask, y_al, cov_AL=None):
    """Compute loss for matrix completion problem"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    loss = torch.norm((cov_O_inverse + z @ z.T)[torch.BoolTensor(mask)]) ** 2

    if y_al is not None:
        int_cov = calculate_cov_OS(z, cov_S, cov_O)
        int_mu = calculate_mu(int_cov, E_S, E_O).clamp(1e-6, 1.0).detach().numpy()
        _, int_probs = predict(label_matrix, int_mu)
        loss += loss_pk_probs(int_probs, y_al, label_matrix)

    if cov_AL is not None:
        # Add loss for current covariance if taking active learning weak label into account
        int_cov = calculate_cov_OS(z, cov_S, cov_O)
        loss += loss_pk_cov(int_cov, cov_AL)

    return loss


def create_mask(cliques, nr_wl, y_dim):
    """Create mask to encode graph structure in covariance matrix"""

    mask = np.ones((nr_wl * y_dim, nr_wl * y_dim))
    for i in range(nr_wl):
        # Mask out diagonal blocks for the individual weak labels
        mask[i * y_dim:(i + 1) * y_dim, i * y_dim:(i + 1) * y_dim] = 0

    # Mask out interactions within cliques
    for clique in cliques:
        for pair in itertools.permutations(clique, r=2):
            i = pair[0]
            j = pair[1]
            mask[i * y_dim:(i + 1) * y_dim, j * y_dim:(j + 1) * y_dim] = 0

    return mask


def fit(label_matrix, al, z, y_al, cliques, class_balance, n_epochs, lr):
    """Fit label model"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    # Transform label matrix to indicator variables
    psi = ((y_set == label_matrix[..., None]) * 1).reshape(N_total, -1)

    # Compute observed expectations and covariances
    E_O = psi.mean(axis=0)
    cov_O = torch.Tensor(np.cov(psi.T, bias=True))
    cov_O_inverse = torch.pinverse(cov_O)

    E_S = (y_set * class_balance).sum()
    cov_Y = np.diag(class_balance) - class_balance.reshape((-1, 1)) @ class_balance.reshape((1, -1))
    # In the rank-one setting we only consider one column of psi(Y)
    cov_S = cov_Y[-1, -1]

    mask = create_mask(cliques, nr_wl, y_dim)

    if al:
        # Calculate known covariance for active learning weak label
        # cov_AL = torch.Tensor(
        #     np.diag(label_matrix_onehot[:, -y_dim:].sum(axis=0) / (np.array(class_balance) * N_total) * cov_S))
        # cov_AL[0, 1] = -cov_AL[0, 0]
        # cov_AL[1, 0] = -cov_AL[1, 1]
        # al_idx = wl_idx[str(label_matrix.shape[1]-1)]
        cov_AL = torch.Tensor(
            (psi[:, -y_dim:] * psi[:, -y_dim:]).mean(axis=0) / class_balance.reshape(-1, 1) * cov_Y)

        # Mask out active learning weak label because it is conditionally dependent on the other weak labels
        mask[:, -y_dim:] = 0
        mask[-y_dim:, :] = 0
    else:
        cov_AL = None

    # if z is None:
    z = nn.Parameter(torch.normal(0, 1, size=(y_dim * nr_wl, y_dim - 1)), requires_grad=True)
    optimizer = torch.optim.Adam({z}, lr=lr)

    # Find optimal z
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_func(z, label_matrix, cov_S, cov_O, cov_O_inverse, E_S, E_O, mask, y_al, cov_AL=cov_AL)
        loss.backward()
        optimizer.step()

    # Compute covariances and label model probabilities from optimal z
    cov_OS = calculate_cov_OS(z, cov_S, cov_O)
    mu = calculate_mu(cov_OS, E_S, E_O).clamp(1e-6, 1.0).detach().numpy()

    return mu, z


def predict(label_matrix, mu):
    """Predict training labels"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    psi = ((y_set == label_matrix[..., None]) * 1).reshape(N_total, -1)

    X_Z = np.zeros((N_total, y_dim))
    for i in range(y_dim):
        # Product of conditional probabilities of weak label votes per data point
        X_Z[:, i] = np.prod(np.tile(mu[:, i], (N_total, 1)), axis=1, where=(psi == 1.0))

    # Normalize to make probability distribution
    Z = X_Z.sum(axis=1)
    probs = X_Z / np.tile(Z, (y_dim, 1)).T

    return np.argmax(np.around(probs), axis=-1), probs


def fit_predict_lm(label_matrix, y_al, label_model_kwargs, al=False, z=None):
    """Fit label model and predict"""

    mu_hat, z = fit(label_matrix, al, z, y_al, **label_model_kwargs)
    Y_hat, probs = predict(label_matrix, mu_hat)

    return Y_hat, probs, z


def get_conditional_probabilities(label_matrix, mu):
    """Get conditional probabilities from label model parameters"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    c_probs = np.zeros(((nr_wl) * (y_dim + 1), y_dim))
    for wl in range(nr_wl):
        # Conditional probabilities are label model parameters
        c_probs[(wl * y_dim) + wl + 1:y_dim + (wl * y_dim) + wl + 1, :] = mu[(wl * y_dim):y_dim + (wl * y_dim), :]

        # Probability for abstain
        c_probs[(wl * y_dim) + wl, :] = 1 - mu[(wl * y_dim):y_dim + (wl * y_dim), :].sum(axis=0)

    return c_probs


def get_accuracies(label_matrix, c_probs, class_balance):
    """Get weak label accuracies"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    # Joint probabilities from conditional
    exp_S = (y_set * class_balance).sum()
    P_Ylam = c_probs * np.tile(exp_S, ((y_dim + 1) * nr_wl, 1))

    weights = np.zeros((1, nr_wl))
    for i in range(nr_wl):
        # Sum of probabilities where weak label and Y agree
        weights[:, i] = P_Ylam[i * (y_dim + 1) + 1, 0] + P_Ylam[i * (y_dim + 1) + 2, 1]

    # Label propensity of weak labels
    coverage = (label_matrix != -1).mean(axis=0)

    return np.clip(weights / coverage, 1e-6, 1)


def get_overall_accuracy(probs, y):
    """Compute overall accuracy from label predictions"""

    return (np.argmax(np.around(probs), axis=-1) == np.array(y)).sum() / len(y)


def get_true_accuracies(label_matrix, y):
    """Obtain actual weak label accuracies from data and ground truth labels"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    coverage = (label_matrix != -1).mean(axis=0)

    true_accuracies = np.zeros((1, nr_wl))
    for i in range(nr_wl):
        true_accuracies[:, i] = (label_matrix[:, i] == y).mean()

    return true_accuracies/coverage


def get_true_mu(label_matrix, y):
    """Obtain actual label model parameters from data and ground truth labels"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    exp_mu = np.zeros((nr_wl * y_dim, y_dim))
    for i in range(y_dim):
        for j in range(y_dim):
            row_ind = list(range(j, y_dim * nr_wl + j, y_dim))
            mean = (label_matrix[y == i] == j).mean(axis=0)
            exp_mu[row_ind, i] = mean

    return exp_mu


# def loss_probs(mu, t: float = 10000) -> torch.Tensor:
#     m_zeros = torch.zeros(mu.shape)
#     m_ones = torch.ones(mu.shape)
#     return t * (torch.norm((m_zeros - mu)[mu < 0]) ** 2 + torch.norm((mu - m_ones)[mu > 1]) ** 2)


# def loss_pk_mu(mu, s: float = 1e1) -> torch.Tensor:
#     r"""Loss from prior knowledge"""
#     O_dim, y_dim = mu.shape
#
#     m = torch.zeros([O_dim, y_dim])
#     m[:-mu.shape[1]] = 1
#
#     mu_known = torch.zeros(mu.shape)
#     for i, k in enumerate(range(O_dim - y_dim, O_dim)):
#         mu_known[k, i] = 1
#
#     return s * torch.norm((mu - mu_known)[m.type(torch.BoolTensor)]) ** 2


# import numpy as np
# import itertools
# import torch
# import torch.nn as nn
# import pandas as pd

# def get_properties(label_matrix):
#     """Get properties such as dimensions from label matrix"""

#     N, nr_wl = label_matrix.shape
#     y_set = np.unique(label_matrix)  # array of classes
#     if y_set[0] == -1:
#         y_set = y_set[1:]
#     y_dim = len(y_set)  # number of classes

#     return N, nr_wl, y_set, y_dim


# def calculate_mu(cov_OS, E_S, E_O):
#     """Compute mu from OS covariance"""

#     return (cov_OS + torch.Tensor(E_O.reshape(-1, 1) @ E_S.reshape(1, -1))) / torch.Tensor(
#         np.tile(E_S, (cov_OS.shape[0], 1)))


# def calculate_cov_OS(z, cov_S, cov_O):
#     """Compute unobserved part of covariance"""

#     c = 1 / cov_S * (1 + torch.mm(torch.mm(z.T, cov_O), z))
#     cov_OS = torch.mm(cov_O, z / torch.sqrt(c))

#     # Add covariance for opposite label
#     if cov_OS[0] < 0:
#         joint_cov_OS = torch.cat((-1 * cov_OS, cov_OS), axis=1)
#     else:
#         joint_cov_OS = torch.cat((cov_OS, -1 * cov_OS), axis=1)

#     return joint_cov_OS


# def loss_prior_knowledge(cov_OS, al_idx, cov_AL, penalty_strength: float = 3):
#     """Compute loss from prior knowledge on part of covariance matrix"""

#     cov_OS_al = cov_OS[al_idx, :]

#     return penalty_strength * torch.norm(cov_OS_al - cov_AL) ** 2


# def loss_func(z, cov_S, cov_O, cov_O_inverse, mask, al_idx=None, cov_AL=None):
#     """Compute loss for matrix completion problem"""

#     loss = torch.norm((cov_O_inverse + z @ z.T)[torch.BoolTensor(mask)]) ** 2

#     if al_idx is not None:
#         # Add loss for current covariance if taking active learning weak label into account
#         tmp_cov_OS = calculate_cov_OS(z, cov_S, cov_O)
#         loss += loss_prior_knowledge(tmp_cov_OS, al_idx, cov_AL)

#     return loss


# def create_mask(cliques, nr_wl, y_dim):
#     """Create mask to encode graph structure in covariance matrix"""

#     mask = np.ones((nr_wl * y_dim, nr_wl * y_dim))
#     for i in range(nr_wl):
#         # Mask out diagonal blocks for the individual weak labels
#         mask[i * y_dim:(i + 1) * y_dim, i * y_dim:(i + 1) * y_dim] = 0

#     # Mask out interactions within cliques
#     for clique in cliques:
#         for pair in itertools.permutations(clique, r=2):
#             i = pair[0]
#             j = pair[1]
#             mask[i * y_dim:(i + 1) * y_dim, j * y_dim:(j + 1) * y_dim] = 0

#     return mask


# def create_mask_int(cliques, nr_wl, y_dim, wl_idx):
#     """Create mask to encode graph structure in covariance matrix"""

#     mask = np.ones((max(max(wl_idx.values()))+1, max(max(wl_idx.values()))+1))

#     for key in wl_idx.keys():
#         mask[wl_idx[key][0]: wl_idx[key][-1] + 1, wl_idx[key][0]: wl_idx[key][-1] + 1] = 0

#         key = key.split("_")

#         # Create all possible subsets of clique
#         clique_list = list(itertools.chain.from_iterable(
#             itertools.combinations(key, r) for r in range(len(key) + 1) if r > 0))

#         # Create all pairs of subsets of clique
#         clique_pairs = list(itertools.permutations(["_".join(clique) for clique in clique_list], r=2))

#         for pair in clique_pairs:
#             i = wl_idx[pair[0]]
#             j = wl_idx[pair[1]]
#             mask[i[0]:i[-1]+1, j[0]:j[-1]+1] = 0

#     return mask


# def get_psi(label_matrix):

#     N_total, _, y_set, _ = get_properties(label_matrix)
#     psi = ((y_set == label_matrix[..., None]) * 1).reshape(N_total, -1)

#     return psi


# def get_psi_int(label_matrix, cliques):

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     psi_list = []
#     col_counter = 0
#     wl_idx = {}
#     for i in range(nr_wl):
#         wl = label_matrix[:, i]
#         # if i == nr_wl - 1:
#         #     weak_label_classes = np.append(-1, y_set)
#         # else:
#         #     weak_label_classes = np.unique(wl)
#         # else:
#         #     # Leave out abstain because it is implicit in onehot-format
#         #     weak_label_classes = np.array([label for label in np.unique(wl) if label != -1])
#         wl_onehot = (wl[:, np.newaxis] == y_set)*1
#         psi_list.append(wl_onehot)
#         wl_idx[str(i)] = list(range(col_counter, col_counter+wl_onehot.shape[1]))
#         col_counter += wl_onehot.shape[1]

#     psi = np.hstack(psi_list)

#     psi_int_list = []
#     clique_idx = {}
#     for clique in cliques:
#         clique_comb = itertools.chain.from_iterable(
#             itertools.combinations(clique, r) for r in range(len(clique)+1) if r > 1)
#         for i, comb in enumerate(clique_comb):
#             if len(comb) == 2:
#                 idx1 = wl_idx[str(comb[0])]
#                 idx2 = wl_idx[str(comb[1])]
#                 wl_int_onehot = (
#                     (psi[:, np.newaxis, idx1[0]:(idx1[-1]+1)]
#                         * psi[:, idx2[0]:(idx2[-1]+1), np.newaxis]).reshape(len(psi), -1)
#                 )

#                 psi_int_list.append(wl_int_onehot)
#                 clique_idx[comb] = i
#                 wl_idx[str(comb[0]) + "_" + str(comb[1])] = list(range(col_counter, col_counter+wl_int_onehot.shape[1]))

#             if len(comb) == 3:
#                 idx3 = wl_idx[str(comb[2])]
#                 wl_int_onehot = (
#                     (psi_int_list[clique_idx[(comb[0], comb[1])]][:, np.newaxis, :]
#                         * psi[:, idx3[0]:(idx3[-1]+1), np.newaxis]).reshape(len(psi), -1)
#                 )
#                 psi_int_list.append(wl_int_onehot)
#                 wl_idx[str(comb[0]) + "_" + str(comb[1]) + "_" + str(comb[2])] = list(
#                     range(col_counter, col_counter+wl_int_onehot.shape[1]))

#             col_counter += wl_int_onehot.shape[1]

#     psi_2 = np.hstack(psi_int_list)

#     return np.concatenate([psi, psi_2], axis=1), wl_idx


# def fit(label_matrix: np.array,
#         cliques: list,
#         class_balance: np.array,
#         z=None,
#         add_cliques: bool = True,
#         active_learning: bool = False,
#         n_epochs: int = 300,
#         lr: float = 1e-1):
#     """Fit label model"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     # Transform label matrix to indicator variables
#     if add_cliques:
#         psi, wl_idx = get_psi_int(label_matrix, cliques)
#     else:
#         psi = get_psi(label_matrix)
#         wl_idx = {}
#         wl_idx[str(label_matrix.shape[1]-1)] = [y_dim*(nr_wl - 1), y_dim*(nr_wl - 1) + 1]

#     # Compute observed expectations and covariances
#     E_O = psi.mean(axis=0)
#     cov_O = torch.Tensor(np.cov(psi.T, bias=True))
#     cov_O_inverse = torch.pinverse(cov_O)

#     E_S = (y_set * class_balance).sum()
#     cov_Y = np.diag(class_balance) - class_balance.reshape((-1, 1)) @ class_balance.reshape((1, -1))
#     # In the rank-one setting we only consider one column of psi(Y)
#     cov_S = cov_Y[-1, -1]

#     if add_cliques:
#         mask = create_mask_int(cliques, nr_wl, y_dim, wl_idx)
#     else:
#         mask = create_mask(cliques, nr_wl, y_dim)

#     if active_learning:
#         # Calculate known covariance for active learning weak label
#         al_idx = wl_idx[str(label_matrix.shape[1]-1)]
#         cov_AL = torch.Tensor(
#             (psi[:, al_idx] * psi[:, al_idx]).mean(axis=0) / class_balance.reshape(-1, 1) * cov_Y)
#     else:
#         al_idx = None
#         cov_AL = None

#     if z is None:
#         z = nn.Parameter(torch.normal(0, 1, size=(psi.shape[1], y_dim - 1)), requires_grad=True)
#     optimizer = torch.optim.Adam({z}, lr=lr)

#     # Find optimal z
#     for epoch in range(n_epochs):
#         optimizer.zero_grad()
#         loss = loss_func(z, cov_S, cov_O, cov_O_inverse, mask, al_idx=al_idx, cov_AL=cov_AL)
#         loss.backward()
#         optimizer.step()

#     # Compute covariances and label model probabilities from optimal z
#     cov_OS = calculate_cov_OS(z, cov_S, cov_O)
#     mu = calculate_mu(cov_OS, E_S, E_O).clamp(1e-6, 1.0).detach().numpy()

#     return mu, z


# def predict_int(label_matrix, mu, cliques, class_balance):
#     """Predict training labels"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     psi, wl_idx = get_psi_int(label_matrix, cliques)

#     cliques_joined = cliques.copy()
#     for i, clique in enumerate(cliques_joined):
#         cliques_joined[i] = ["_".join(str(wl) for wl in clique)]
#     idx = np.array([idx for clique in cliques_joined for i, idx in enumerate(wl_idx[clique[0]])])

#     X_Z = np.zeros((N_total, y_dim))
#     for i in range(y_dim):
#         # Product of conditional probabilities of weak label votes per data point
#         X_Z[:, i] = np.prod(np.tile(mu[idx, i], (N_total, 1)), axis=1, where=(psi[:, idx] == 1)) / class_balance[i]

#     # Normalize to make probability distribution
#     Z = X_Z.sum(axis=1)
#     probs = X_Z / np.tile(Z, (y_dim, 1)).T

#     return np.argmax(np.around(probs), axis=-1), probs


# def predict(label_matrix, mu, cliques, class_balance):
#     """Predict training labels"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     psi = ((y_set == label_matrix[..., None]) * 1).reshape(N_total, -1)
#     E_O = psi.mean(axis=0)

#     X_Z = np.zeros((N_total, y_dim))
#     for i in range(y_dim):
#         # Product of conditional probabilities of weak label votes per data point
#         X_Z[:, i] = np.prod(np.tile(mu[:, i], (N_total, 1)), axis=1, where=(psi[:, :] == 1)) / class_balance[i]# / np.prod(np.tile(E_O[:], (N_total, 1)), axis=1, where=(psi == 1))

#     # # Normalize to make probability distribution
#     Z = X_Z.sum(axis=1)
#     probs = X_Z / np.tile(Z, (y_dim, 1)).T

#     return np.argmax(np.around(probs), axis=-1), probs


# def fit_predict_lm(label_matrix, cliques, class_balance, label_model_kwargs, active_learning=False, add_cliques=False, z=None):
#     """Fit label model and predict"""

#     mu_hat, z = fit(label_matrix, cliques, class_balance, z, add_cliques, active_learning, **label_model_kwargs)

#     if add_cliques:
#         Y_hat, probs = predict_int(label_matrix, mu_hat, cliques, class_balance)
#     else:
#         Y_hat, probs = predict(label_matrix, mu_hat, cliques, class_balance)

#     return Y_hat, probs, z


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


# def get_overall_accuracy(probs, y):
#     """Compute overall accuracy from label predictions"""

#     return (np.argmax(np.around(probs), axis=-1) == np.array(y)).sum() / len(y)


# def get_true_accuracies(label_matrix, y):
#     """Obtain actual weak label accuracies from data and ground truth labels"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     coverage = (label_matrix != -1).mean(axis=0)

#     true_accuracies = np.zeros((1, nr_wl))
#     for i in range(nr_wl):
#         true_accuracies[:, i] = (label_matrix[:, i] == y).mean()

#     return true_accuracies/coverage


# def get_true_mu(label_matrix, y):
#     """Obtain actual label model parameters from data and ground truth labels"""

#     N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

#     exp_mu = np.zeros((nr_wl * y_dim, y_dim))
#     for i in range(y_dim):
#         for j in range(y_dim):
#             row_ind = list(range(j, y_dim * nr_wl + j, y_dim))
#             mean = (label_matrix[y == i] == j).mean(axis=0)
#             exp_mu[row_ind, i] = mean

#     return exp_mu


# # def loss_probs(mu, t: float = 10000) -> torch.Tensor:
# #     m_zeros = torch.zeros(mu.shape)
# #     m_ones = torch.ones(mu.shape)
# #     return t * (torch.norm((m_zeros - mu)[mu < 0]) ** 2 + torch.norm((mu - m_ones)[mu > 1]) ** 2)


# # def loss_pk_mu(mu, s: float = 1e1) -> torch.Tensor:
# #     r"""Loss from prior knowledge"""
# #     O_dim, y_dim = mu.shape
# #
# #     m = torch.zeros([O_dim, y_dim])
# #     m[:-mu.shape[1]] = 1
# #
# #     mu_known = torch.zeros(mu.shape)
# #     for i, k in enumerate(range(O_dim - y_dim, O_dim)):
# #         mu_known[k, i] = 1
# #
# #     return s * torch.norm((mu - mu_known)[m.type(torch.BoolTensor)]) ** 2
