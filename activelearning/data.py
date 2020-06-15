import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sample_X(N_1, N_2, centroids, std):

    x1_1 = np.random.normal(loc=centroids[0,0], scale=std, size=N_1).reshape(-1, 1)
    x2_1 = np.random.normal(loc=centroids[0,1], scale=std, size=N_1).reshape(-1, 1)
    x1_2 = np.random.normal(loc=centroids[1,0], scale=std, size=N_2).reshape(-1, 1)
    x2_2 = np.random.normal(loc=centroids[1,1], scale=std, size=N_2).reshape(-1, 1)

    X_1 = np.concatenate([x1_1, x2_1], axis=1)
    X_2 = np.concatenate([x1_2, x2_2], axis=1)
    X = np.concatenate([X_1, X_2])

    return X


def compute_decision_boundary(centroids):

    # Point in the middle of cluster means
    bp1 = centroids.sum(axis=0) / 2

    # Vector between cluster means
    diff = centroids[1, :] - centroids[0, :]

    # Slope of decision boundary is perpendicular to slope of line that goes through the cluster means
    slope = diff[1] / diff[0]
    perp_slope = -1 / slope

    # Solve for intercept using slope and middle point
    b = bp1[1] - perp_slope * bp1[0]

    return b, perp_slope


def sample_y(centroids, X, N, scaling_factor):

    b, perp_slope = compute_decision_boundary(centroids)

    coef = [b, perp_slope, -1]
    coef = [scaling_factor * co for co in coef]

    Z = coef[0] + coef[1] * X[:, 0] + coef[2] * X[:, 1]

    p = sigmoid(Z)

    y = np.random.binomial(1, p, N)

    return X, p, y


def concat_centroids(centroid_1, centroid_2):
    return np.concatenate([centroid_1.reshape(1, -1), centroid_2.reshape(1, -1)], axis=0)


def sample_clusters(N_1, N_2, centroid_1, centroid_2, std = 0.5, scaling_factor = 1):

    N_total = N_1 + N_2

    centroids = concat_centroids(centroid_1, centroid_2)

    X = sample_X(N_1, N_2, centroids, std)

    return sample_y(centroids, X, N_total, scaling_factor)
