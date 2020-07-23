import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

from label_model import fit_predict_lm, get_overall_accuracy
from final_model import fit_predict_fm


def AL_query(probs, label_matrix, y_al):
    """Choose data point to label from label predictions"""

    nr_wl = label_matrix.shape[1]

    abs_diff = np.abs(probs[:, 1] - probs[:, 0])

    # Find data points the model where the model is least confident
    minimum = min(j for i, j in enumerate(abs_diff) if y_al[i] == -1)
    indices = [j for j, v in enumerate(abs_diff) if v == minimum]

    # Make really random
    random.seed(random.SystemRandom().random())

    # Pick a random point from least confident data points
    return random.choice(indices), y_al


def random_query(label_matrix):

    nr_wl = label_matrix.shape[1]

    indices = [j for j, v in enumerate(range(label_matrix.shape[0])) if label_matrix[j, nr_wl - 1] == -1]

    random.seed(random.SystemRandom().random())

    # Pick a random point
    return random.choice(indices)


def AL_fm_query(lm_probs, fm_probs, label_matrix):
    """Choose data point to label from label predictions"""

    abs_diff_lm = np.abs(lm_probs[:, 1] - lm_probs[:, 0])
    abs_diff_fm = np.abs(fm_probs[:, 1] - fm_probs[:, 0])

    # Find data points the model where the model is least confident
    # minimum = np.where(abs_diff_lm == abs_diff_lm.min())
    # index = np.where(abs_diff_fm == min([j for i, j in enumerate(abs_diff_fm) if i in minimum[0]]))[0][0]

    min_mask = abs_diff_lm == abs_diff_lm.min()
    min_indices = np.where(abs_diff_lm == abs_diff_lm.min())
    # sel = np.where(np.sort(abs_diff_fm[min_mask]) < 0.2)[0][-1]
    indices = min_indices[0][np.argsort(abs_diff_fm[min_mask])[:100]]

    random.seed(random.SystemRandom().random())
    return random.choice(indices)


def AL_pipeline(label_matrix, df, label_model_kwargs, final_model_kwargs, it):
    """Iteratively refine label matrix and training set predictions with active learning strategy"""

    nr_wl = label_matrix.shape[1]

    accuracies = {}
    accuracies["prob_labels"] = []
    accuracies["final_labels"] = []
    queried = []
    probs_dict = {}
    prob_label_dict = {}

    y = df["y"].values
    y_al = np.full_like(y, -1)

    Y_hat, old_probs, z = fit_predict_lm(label_matrix, y_al, label_model_kwargs, al=False, z=None)
    # accuracies["prob_labels"].append(get_overall_accuracy(old_probs, y))

    # _, final_probs = fit_predict_fm(df[["x1", "x2"]].values, old_probs, **final_model_kwargs, soft_labels=True)
    # probs_dict[0] = final_probs[:, 1]
    prob_label_dict[0] = old_probs[:, 1]

    # final_model_kwargs["n_epochs"] = 50

    # accuracies["final_labels"].append(get_overall_accuracy(final_probs, y))

    for i in tqdm(range(it)):
        # sel_idx = AL_fm_query(old_probs, final_probs, label_matrix)
        # sel_idx = random_query(label_matrix)
        sel_idx, y_al = AL_query(old_probs, label_matrix, y_al)

        # print("Iteration:", i + 1, " Label combination", label_matrix[sel_idx, :nr_wl], " True label:", y[sel_idx],
        #       "Estimated label:", Y_hat[sel_idx], " selected index:", sel_idx)

        # Add label to label matrix
        # label_matrix[sel_idx, nr_wl - 1] = y[sel_idx]
        y_al[sel_idx] = y[sel_idx]

        # Fit label model on refined label matrix
        Y_hat, new_probs, z = fit_predict_lm(label_matrix, y_al, label_model_kwargs, al=False, z=z)

        # print("Before:", old_probs[sel_idx, :], "After:", new_probs[sel_idx])

        # accuracies["prob_labels"].append(get_overall_accuracy(new_probs, y))

        # queried.append(sel_idx)

        # if (i+1) % 50 == 0:
        # _, final_probs = fit_predict_fm(df[["x1", "x2"]].values, new_probs, **final_model_kwargs, soft_labels=True)
        # probs_dict[i+1] = final_probs[:, 1]
        prob_label_dict[i+1] = new_probs[:, 1]

        #     accuracies["final_labels"].append(get_overall_accuracy(final_probs, y))

        old_probs = new_probs.copy()

    return new_probs, accuracies, queried, probs_dict, prob_label_dict
