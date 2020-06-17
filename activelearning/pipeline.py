import numpy as np
import random

from label_model import fit_predict_lm, get_overall_accuracy


def AL_query(probs, label_matrix):
    """Choose data point to label from label predictions"""

    nr_wl = label_matrix.shape[1]

    abs_diff = np.abs(probs[:, 1] - probs[:, 0])

    # Find data points the model where the model is least confident
    minimum = min(j for i, j in enumerate(abs_diff) if label_matrix[i, nr_wl - 1] == -1)
    indices = [j for j, v in enumerate(abs_diff) if v == minimum]

    # Make really random
    random.seed(random.SystemRandom().random())

    # Pick a random point from least confident data points
    return random.choice(indices)


def AL_pipeline(label_matrix, y, label_model_kwargs, it):
    """Iteratively refine label matrix and training set predictions with active learning strategy"""

    nr_wl = label_matrix.shape[1]

    accuracies = []
    queried = []

    Y_hat, old_probs = fit_predict_lm(label_matrix, label_model_kwargs, al=True)
    accuracies.append(get_overall_accuracy(old_probs, y))

    for i in range(it):
        sel_idx = AL_query(old_probs, label_matrix)

        print("Iteration:", i + 1, " Label combination", label_matrix[sel_idx, :nr_wl], " True label:", y[sel_idx],
              "Estimated label:", Y_hat[sel_idx], " selected index:", sel_idx)

        # Add label to label matrix
        label_matrix[sel_idx, nr_wl - 1] = y[sel_idx]

        # Fit label model on refined label matrix
        Y_hat, new_probs = fit_predict_lm(label_matrix, label_model_kwargs, al=True)

        print("Before:", old_probs[sel_idx, :], "After:", new_probs[sel_idx])

        accuracy = get_overall_accuracy(new_probs, y)
        accuracies.append(accuracy)

        queried.append(sel_idx)

        old_probs = new_probs.copy()

    return new_probs, accuracies, queried
