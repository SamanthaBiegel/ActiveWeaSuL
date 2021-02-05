import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm_notebook as tqdm

from activeweasul.label_model import LabelModel
from activeweasul.plot import PlotMixin
from activeweasul.query import ActiveLearningQuery


def set_seed(seed=42):
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
        query_strategy (str, optional): Active learning query strategy, one of ["maxkl", "margin", "nashaat"]
        randomness (float, optional): Probability of choosing a random point instead of following strategy
        final_model: Optional discriminative model object
        batch_size (int, optional): Batch size if training discriminate model
        discr_model_frequency (int, optional): Interval for training the discriminative model
        starting_seed (int, optional): Seed for first part of pipeline (initial label model)
        seed (int, optional): Seed for remainder of pipeline
    """

    def __init__(self,
                 it: int = 30,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 penalty_strength: float = 1e3,
                 query_strategy: str = "maxkl",
                 randomness: float = 0,
                 final_model=None,
                 batch_size: int = 20,
                 discr_model_frequency: int = 1,
                 starting_seed: int = 76,
                 seed: int = 65):

        super().__init__(query_strategy=query_strategy)

        self.it = it
        self.label_model = LabelModel(n_epochs=n_epochs,
                                      lr=lr,
                                      hide_progress_bar=True)
        self.penalty_strength = penalty_strength
        self.query_strategy = query_strategy
        self.randomness = randomness
        self.final_model = final_model
        self.batch_size = batch_size
        self.discr_model_frequency = discr_model_frequency
        set_seed(starting_seed)
        self.seed = seed

    def run_active_weasul(self, label_matrix, y_train, cliques, class_balance, label_matrix_test=None, y_test=None, train_dataset=None, test_dataset=None):
        """Iteratively label points, refit label model and return adjusted probabilistic labels

        Args:
            label_matrix (numpy.array): Array with labeling function outputs on train set
            y_train (numpy.array): Ground truth labels of training dataset
            cliques (list): List of lists of maximal cliques (column indices of label matrix)
            class_balance (numpy.array): Array with true class distribution
            label_matrix_test (numpy.array): Array with labeling function outputs on test set
            y_test (numpy.array): Ground truth labels of test set
            train_dataset (torch.utils.data.Dataset, optional): Train dataset if training discriminative model on image data. Should be
                custom dataset with attribute Y containing target labels.
            test_dataset (torch.utils.data.Dataset, optional): Test dataset if training discriminative model on image data

        Returns:
            numpy.array: Array with probabilistic labels for training dataset
        """
        # TODO: make predicting on test set optional
        if any(v is None for v in (label_matrix_test, y_test, test_dataset)):
            label_matrix_test = label_matrix.copy()
            y_test = y_train.copy()
            test_dataset = train_dataset

        self.label_matrix = label_matrix.copy()
        self.y_train = y_train
        self.y_test = y_test

        self.ground_truth_labels = np.full_like(y_train, -1)

        dl_test = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)

        # Identify buckets
        self.unique_combs, self.unique_idx, self.unique_inverse = np.unique(label_matrix,
                                                                            return_index=True,
                                                                            return_inverse=True,
                                                                            axis=0)

        # Used for plotting
        self.confs = {range(len(self.unique_idx))[i]:
                      "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}

        for i in tqdm(range(self.it + 1), desc="Active Learning Iterations"):

            # Fit label model and predict to obtain probabilistic labels
            prob_labels_train = self.label_model.fit(label_matrix=self.label_matrix,
                                                     cliques=cliques,
                                                     class_balance=class_balance,
                                                     ground_truth_labels=self.ground_truth_labels).predict()
            prob_labels_test = self.label_model.predict(label_matrix_test,
                                                        self.label_model.mu,
                                                        self.label_model.E_S)

            # Optionally, train discriminative model on probabilistic labels
            if self.final_model is not None and i % self.discr_model_frequency == 0:
                train_dataset.Y = prob_labels_train.clone().detach()
                dl_train = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)

                self.final_model.reset()
                preds_train = self.final_model.fit(dl_train).predict()
                preds_test = self.final_model.predict(dl_test)
            else:
                preds_train = None
                preds_test = None

            if i == 0:
                sel_idx = None
                # Different seed for rest of the pipeline after first label model fit
                set_seed(self.seed)

            self.log(count=i, lm_train=prob_labels_train, lm_test=prob_labels_test, fm_train=preds_train,
                     fm_test=preds_test, selected_point=sel_idx)

            if i == 0:
                # Switch to active learning mode
                self.label_model.active_learning = True
                self.label_model.penalty_strength = self.penalty_strength
                self.ground_truth_labels = np.full_like(y_train, -1)

            if i < self.it:
                # Query point and add to ground truth labels
                sel_idx = self.sample(prob_labels_train)
                self.ground_truth_labels[sel_idx] = self.y_train[sel_idx]

                if self.query_strategy == "nashaat":
                    self.label_model.active_learning = False
                    # Nashaat et al. replace labeling function outputs by ground truth
                    self.label_matrix[sel_idx, :] = self.y_train[sel_idx]

        return prob_labels_train

    def log(self, count, lm_train, lm_test, fm_train, fm_test, selected_point=None):
        """Keep track of performance metrics and label predictions"""

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
        self.probs["bucket_labels_train"][count] = self.probs["Generative_train"][count][self.unique_idx]
        self.mu_dict[count] = self.label_model.mu.clone().detach().numpy().squeeze()

        if self.final_model is not None and count % self.discr_model_frequency == 0:
            self.metrics["Discriminative_train"][count] = self.final_model.analyze(self.y_train, fm_train)
            self.metrics["Discriminative_test"][count] = self.final_model.analyze(self.y_test, fm_test)
            self.probs["Discriminative_train"][count] = fm_train[:, 1].clone().cpu().detach().numpy()
            self.probs["Discriminative_test"][count] = fm_test[:, 1].clone().cpu().detach().numpy()

        if selected_point:
            self.queried.append(selected_point)
            if self.query_strategy in ["maxkl", "margin"] and self.randomness == 0:
                self.bucket_AL_values[count] = self.bucket_values

        return self


class CustomTensorDataset(TensorDataset):
    """Custom Tensor Dataset"""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def update(self, X, Y):
        """Update dataset content

        Args:
            X (torch.Tensor): Tensor with features (columns)
            Y (torch.Tensor): Tensor with labels
        """
        self.X = X
        self.Y = Y
