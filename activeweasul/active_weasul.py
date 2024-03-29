import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

from datasets import CustomTensorDataset
from label_model import LabelModel
from plot import PlotMixin
from query import ActiveLearningQuery


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ActiveWeaSuLPipeline(PlotMixin, ActiveLearningQuery):
    """Pipeline to run Active WeaSuL.

    Args:
        it (int, optional): Number of active learning iterations
        n_epochs (int, optional): Number of label model epochs
        lr (float, optional): Label model learning rate
        penalty_strength (float, optional): Strength of the active learning penalty
        query_strategy (str, optional): Active learning query strategy, one of
            ["maxkl", "margin", "nashaat"]
        randomness (float, optional): Probability of choosing a random point instead
            of following strategy
        discriminative_model: Optional discriminative model object
        batch_size (int, optional): Batch size if training discriminate model
        discr_model_frequency (int, optional): Interval for training the discriminative model
        starting_seed (int, optional): Seed for first part of pipeline (initial label model)
        seed (int, optional): Seed for remainder of pipeline
    """

    def __init__(
        self, it: int = 30, n_epochs: int = 200, lr: float = 1e-1, penalty_strength: float = 1e3,
        query_strategy: str = "maxkl", randomness: float = 0, discriminative_model=None, batch_size: int = 20,
            discr_model_frequency: int = 1, starting_seed: int = 76, seed: int = 65):

        super().__init__(query_strategy=query_strategy)

        self.it = it
        self.label_model = LabelModel(
            n_epochs=n_epochs, lr=lr, hide_progress_bar=True)
        self.penalty_strength = penalty_strength
        self.query_strategy = query_strategy
        self.randomness = randomness
        self.discriminative_model = discriminative_model
        self.batch_size = batch_size
        self.discr_model_frequency = discr_model_frequency
        set_seed(starting_seed)
        if discriminative_model is not None:
            self.discriminative_model.reset()
            self.discriminative_model.min_val_loss = 1e15
        self.seed = seed

    def run_active_weasul(
        self, label_matrix: np.ndarray, y_train: np.ndarray, cliques: list,
        class_balance: np.ndarray, label_matrix_test: np.ndarray = None,
        y_test: np.ndarray = None, train_dataset: torch.utils.data.Dataset = None,
            test_dataset: torch.utils.data.Dataset = None):
        """Iteratively label points, refit label model and return adjusted probabilistic labels.

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on train set
            y_train (numpy.array): Ground truth labels of training dataset
            cliques (list): List of lists of maximal cliques (column indices of label matrix)
            class_balance (numpy.array): Array with true class distribution
            label_matrix_test (numpy.array, optional): Array with labeling function outputs on test set
            y_test (numpy.array, optional): Ground truth labels of test set
            train_dataset (torch.utils.data.Dataset, optional): Train dataset if training
                discriminative model on image data. Should be
                custom dataset with attribute Y containing target labels.
            test_dataset (torch.utils.data.Dataset, optional): Test dataset if training
                discriminative model on image data

        Returns:
            torch.Tensor: Tensor with probabilistic labels for training dataset
        """
        if any(v is None for v in (label_matrix_test, y_test, test_dataset)):
            label_matrix_test = label_matrix.copy()
            y_test = y_train.copy()
            test_dataset = train_dataset

        self.label_matrix = label_matrix.copy()
        self.label_matrix_test = label_matrix_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        self.ground_truth_labels = np.full_like(y_train, -1)

        dl_test = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)

        if self.discriminative_model is not None and self.discriminative_model.early_stopping:
            # Split into train and validation sets for early stopping
            indices_shuffle = np.random.permutation(len(self.label_matrix))
            split_nr = int(np.ceil(0.9 * len(self.label_matrix)))
            self.train_idx, val_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]
        else:
            self.train_idx = range(len(self.y_train))

        # Identify buckets
        self.unique_combs, self.unique_idx, self.unique_inverse = np.unique(
            label_matrix, return_index=True, return_inverse=True, axis=0)
        self.bucket_conf_dict = {range(len(self.unique_idx))[i]:
                      "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}

        for i in range(self.it + 1):

            # Fit label model and predict to obtain probabilistic labels
            prob_labels_train = self.label_model.fit(
                label_matrix=self.label_matrix, cliques=cliques,
                class_balance=class_balance, ground_truth_labels=self.ground_truth_labels
            ).predict()
            prob_labels_test = self.label_model.predict(
                self.label_matrix_test, self.label_model.mu, self.label_model.E_S)

            # Optionally, train discriminative model on probabilistic labels
            if self.discriminative_model is not None and i % self.discr_model_frequency == 0:
                discriminative_model_probs_train = prob_labels_train.clone().detach()
                # Replace probabilistic labels with ground truth for labelled points
                discriminative_model_probs_train[self.ground_truth_labels == 1, :] = (
                    torch.DoubleTensor([0, 1]))
                discriminative_model_probs_train[self.ground_truth_labels == 0, :] = (
                    torch.DoubleTensor([1, 0]))
                train_dataset.Y = discriminative_model_probs_train

                if i > 0:
                    # Reset discriminative model parameters to train with updated labels
                    self.discriminative_model.reset()
                dl_train = DataLoader(
                        CustomTensorDataset(*train_dataset[self.train_idx]),
                        shuffle=True, batch_size=self.batch_size)
                if self.discriminative_model.early_stopping:
                    dl_val = DataLoader(
                        CustomTensorDataset(*train_dataset[val_idx]),
                        shuffle=True, batch_size=self.batch_size)
                else:
                    dl_val = None
                preds_train = self.discriminative_model.fit(dl_train, dl_val).predict()
                preds_test = self.discriminative_model.predict(dl_test)
            else:
                preds_train = None
                preds_test = None

            if i == 0:
                sel_idx = None
                # Different seed for rest of the pipeline after first label model fit
                set_seed(self.seed)

                # Switch to active learning mode
                self.label_model.active_learning = True
                self.label_model.penalty_strength = self.penalty_strength

            self.log(
                count=i, lm_train=prob_labels_train, lm_test=prob_labels_test,
                fm_train=preds_train, fm_test=preds_test, selected_point=sel_idx)

            if i < self.it:
                # Query point and add to ground truth labels
                sel_idx = self.sample(prob_labels_train)
                self.ground_truth_labels[sel_idx] = self.y_train[sel_idx]

                if self.query_strategy == "nashaat":
                    self.label_model.active_learning = False

                    # Nashaat et al. replace labeling function outputs by ground truth
                    self.label_matrix[sel_idx, :] = self.y_train[sel_idx]

        return prob_labels_train

    def log(self, count: int, lm_train, lm_test, fm_train, fm_test, selected_point):
        """Keep track of performance metrics, label predictions and parameter values

        Args:
            count (int): Active learning iteration number
            lm_train (torch.Tensor): Tensor with probabilistic labels for training dataset
            lm_test (torch.Tensor): Tensor with probabilistic labels for test dataset
            fm_train (torch.Tensor): Tensor with discriminative model predictions for training dataset
            fm_test (torch.Tensor): Tensor with discriminative model predictions for test dataset
            selected_point (int): Index of data point selected in current iteration
        """

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
        self.probs["bucket_labels_train"][count] = (
            self.probs["Generative_train"][count][self.unique_idx])
        self.mu_dict[count] = self.label_model.mu.clone().detach().numpy().squeeze()

        if self.discriminative_model is not None and count % self.discr_model_frequency == 0:
            self.metrics["Discriminative_train"][count] = self.discriminative_model.analyze(
                self.y_train[self.train_idx], fm_train)
            self.metrics["Discriminative_test"][count] = self.discriminative_model.analyze(
                self.y_test, fm_test)
            self.probs["Discriminative_train"][count] = (
                fm_train[:, 1].clone().cpu().detach().numpy())
            self.probs["Discriminative_test"][count] = (
                fm_test[:, 1].clone().cpu().detach().numpy())

        if selected_point:
            self.queried.append(selected_point)
            if self.query_strategy in ["maxkl", "margin"] and self.randomness == 0:
                self.bucket_AL_values[count] = self.bucket_values
