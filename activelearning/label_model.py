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


def calculate_cov(z, cov_S, cov_O):
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


def loss_func(z, cov_S, cov_O, mask, al=False, cov_AL=None):
    """Compute loss for matrix completion problem"""

    loss = torch.norm((torch.Tensor(np.linalg.pinv(cov_O)) + z @ z.T)[torch.BoolTensor(mask)]) ** 2

    if al:
        # Add loss for current covariance if taking active learning weak label into account
        int_cov = calculate_cov(z, cov_S, cov_O)
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


def fit(label_matrix, al, cliques, class_balance, n_epochs, lr):
    """Fit label model"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    # Transform label matrix to indicator variables
    label_matrix_onehot = ((y_set == label_matrix[..., None]) * 1).reshape(N_total, -1)

    # Compute observed expectations and covariances
    exp_O = label_matrix_onehot.mean(axis=0)
    cov_O = torch.Tensor(np.cov(label_matrix_onehot.T))

    exp_S = (y_set * class_balance).sum()
    cov_S = (y_set ** 2 * class_balance).sum() - exp_S * exp_S

    mask = create_mask(cliques, nr_wl, y_dim)

    if al:
        # Calculate known covariance for active learning weak label
        cov_AL = torch.Tensor(
            np.diag(label_matrix_onehot[:, -y_dim:].sum(axis=0) / (np.array(class_balance) * N_total) * cov_S))
        cov_AL[0, 1] = -cov_AL[0, 0]
        cov_AL[1, 0] = -cov_AL[1, 1]

        # Mask out active learning weak label because it is conditionally dependent on the other weak labels
        mask[:, -y_dim:] = 0
        mask[-y_dim:, :] = 0
    else:
        cov_AL = None

    z = nn.Parameter(torch.normal(0, 1, size=(y_dim * nr_wl, y_dim - 1)), requires_grad=True)
    optimizer = torch.optim.Adam({z}, lr=lr)

    # Find optimal z
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_func(z, cov_S, cov_O, mask, al, cov_AL=cov_AL)
        loss.backward()
        optimizer.step()

    # Compute covariances and label model probabilities from optimal z
    cov_OS = calculate_cov(z, cov_S, cov_O)
    mu = calculate_mu(cov_OS, exp_S, exp_O).clamp(1e-6, 1.0).detach().numpy()

    return mu


def predict(label_matrix, mu):
    """Predict training labels"""

    N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

    label_matrix_onehot = ((y_set == label_matrix[..., None]) * 1).reshape(N_total, -1)

    X_Z = np.zeros((N_total, y_dim))
    for i in range(y_dim):
        # Product of conditional probabilities of weak label votes per data point
        X_Z[:, i] = np.prod(np.tile(mu[:, i], (N_total, 1)), axis=1, where=(label_matrix_onehot == 1.0))

    # Normalize to make probability distribution
    Z = X_Z.sum(axis=1)
    probs = X_Z / np.tile(Z, (y_dim, 1)).T

    return np.argmax(np.around(probs), axis=-1), probs


def fit_predict_lm(label_matrix, label_model_kwargs, al=False):
    """Fit label model and predict"""

    mu_hat = fit(label_matrix, al, **label_model_kwargs)

    return predict(label_matrix, mu_hat)


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
