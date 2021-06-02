import numpy as np
import pandas as pd
import random
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader

from active_weasul import ActiveWeaSuLPipeline, set_seed
from datasets import CustomTensorDataset
from performance import PerformanceMixin


def active_weasul_experiment(
    nr_trials, al_it, label_matrix, y_train, cliques, class_balance, query_strategy,
    starting_seed=76, seeds=None, discr_model_frequency=1, penalty_strength=1, batch_size=20,
    discriminative_model=None, train_dataset=None, test_dataset=None, label_matrix_test=None,
        y_test=None, randomness=0):
    """Run Active WeaSuL with given settings for a number of times.

    Args:
        nr_trials (int): Number of experiment runs
        al_it (int): Number of active learning iterations
        label_matrix (numpy.array): Array with labeling function outputs on train set
        y_train (numpy.array): Ground truth labels of training dataset
        cliques (list): List of lists of maximal cliques (column indices of label matrix)
        class_balance (numpy.array): Array with true class distribution
        query_strategy (str, optional): Active learning query strategy, one of
            ["maxkl", "margin", "nashaat"]
        starting_seed (int, optional): Seed for first part of pipeline (initial label model)
        seeds (numpy.array, optional): Array with seeds for remainder of pipeline
        discr_model_frequency (int, optional): Interval for training the discriminative model. Defaults to 1
        penalty_strength (float, optional): Strength of the active learning penalty. Defaults to 1
        batch_size (int, optional): Batch size if training discriminate model
        discriminative_model: Optional discriminative model object
        train_dataset (torch.utils.data.Dataset, optional): Train dataset if training
                discriminative model on image data. Should be
                custom dataset with attribute Y containing target labels
        test_dataset (torch.utils.data.Dataset, optional): Test dataset if training
                discriminative model on image data
        label_matrix_test (numpy.array): Array with labeling function outputs on test set
        y_test (numpy.array): Ground truth labels of test set
        randomness (float, optional): Probability of choosing a random point instead
            of following strategy

    Returns:
        dict: Dictionary of trials, active learning iterations and metrics
        dict: Dictionary of trials, active learning iterations and entropies
    """

    al_metrics = {}
    al_entropies = {}

    if seeds is None:
        seeds = np.random.randint(0, 1000, nr_trials)

    for i in range(nr_trials):
        seed = seeds[i]

        # Initialize Active WeaSuL pipeline
        al = ActiveWeaSuLPipeline(
            it=al_it,
            penalty_strength=penalty_strength,
            query_strategy=query_strategy,
            randomness=randomness,
            discriminative_model=discriminative_model,
            batch_size=batch_size,
            discr_model_frequency=discr_model_frequency,
            starting_seed=starting_seed,
            seed=seed)

        # Run Active WeaSuL pipeline
        _ = al.run_active_weasul(
            label_matrix=label_matrix,
            y_train=y_train,
            cliques=cliques,
            class_balance=class_balance,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            label_matrix_test=label_matrix_test,
            y_test=y_test)

        al_metrics[i] = al.metrics
        
        # Compute entropies for active learning iterations
        al_entropies[i] = []
        for j in range(al_it):
            bucket_list = al.unique_inverse[al.queried[:j + 1]]
            al_entropies[i].append(
                entropy([
                    len(np.where(bucket_list == j)[0]) / len(bucket_list)
                    for j in range(len(np.unique(al.unique_inverse)))
                ]))

    return al_metrics, al_entropies


def process_metric_dict(metric_dict, strategy_string):
    """Process dictionary of Active WeaSuL metrics

    Args:
        metric_dict (dict): Dictionary of active learning metrics and metrics
        strategy_string (string): Approach corresponding to results in dictionary input

    Returns:
        pandas.DataFrame: Processed dataframe with metrics per trial
    """

    metric_df = (
        pd.DataFrame(metric_dict)
        .stack()
        .apply(pd.Series)
        .stack(dropna=False)
        .reset_index()
        .rename(columns={
            "level_0": "Number of labeled points",
            "level_1": "Model",
            "level_2": "Metric", 0: "Value"
        })
    )
    metric_df[["Model", "Set"]] = metric_df["Model"].str.split("_", expand=True)
    metric_df["Approach"] = strategy_string

    return metric_df


def process_exp_dict(exp_dict, strategy_string):
    """Process dictionary of trial metrics from Active WeaSuL experiments

    Args:
        exp_dict (dict): Dictionary of trials, active learning iterations and metrics
        strategy_string (string): Approach corresponding to results in dictionary input

    Returns:
        pandas.DataFrame: Processed dataframe with metrics per trial
    """

    exp_df = pd.concat({
        i: process_metric_dict(exp_dict[i], strategy_string).drop(index=1)
        for i in exp_dict.keys()
    })
    return exp_df


def process_entropies(entropy_dict, approach_string):
    """Process dictionary with entropy data from Active WeaSuL to dataframe

    Args:
        entropy_dict (dict): Dictionary of trials, active learning iterations and entropies
        approach_string (string): Approach corresponding to results in dictionary input

    Returns:
        pandas.DataFrame: Processed dataframe with entropies
    """

    entropies_df = pd.DataFrame.from_dict(entropy_dict).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
    entropies_df["Approach"] = approach_string

    entropies_df["Number of labeled points"] = entropies_df["Number of labeled points"].apply(lambda x: x+1)
    entropies_df = entropies_df[entropies_df["Number of labeled points"] < 51]

    return entropies_df


def add_weak_supervision_baseline(metric_dfs, al_it):
    """Add weak supervision baseline metrics based on Active WeaSuL results at iteration 0

    Args:
        metric_dfs (pandas.DataFrame): Processed dataframe with metrics. Should have
            at least Active WeaSuL results
        al_it (int): Number of active learning iterations

    Returns:
        pandas.DataFrame: Given metric dataframe with added weak supervision baseline
    """

    baseline_df = (
        pd.concat([
            metric_dfs[
                (metric_dfs["Number of labeled points"] == 0) &
                (metric_dfs["Approach"] == "Active WeaSuL")
            ].groupby(["Metric", "Set", "Model"])
            .mean()
            .reset_index()
        ] * (al_it + 1), ignore_index=True)
    )

    idx = baseline_df.index
    baseline_df["Number of labeled points"] = idx[:al_it + 1].repeat(len(baseline_df) / (al_it + 1))
    baseline_df["Approach"] = "Weak supervision by itself"
    baseline_df["Run"] = 0

    return pd.concat([metric_dfs, baseline_df])


def query_margin(preds, is_in_pool):
    margin = torch.abs(preds[:, 1] - preds[:, 0])
    minimum = torch.min(margin[is_in_pool]).item()
    chosen_points = torch.where((margin == minimum) & (is_in_pool))[0]
    point_idx = torch.randint(0, len(chosen_points), (1, 1))
    point = chosen_points[point_idx].item()
    is_in_pool[point] = False
    return point, is_in_pool


def active_learning_experiment(
    nr_trials, al_it, train_features, y_train, y_test, seeds,
        test_features, batch_size=20, input_model=None, solver="lbfgs"):
    """Run active learning for a number of times

    Args:
        nr_trials (int): Number of experiment runs
        al_it (int): Number of active learning iterations
        train_features (torch.Tensor): Training dataset features
        y_train (numpy.array): Ground truth labels of training dataset
        y_test (numpy.array): Ground truth labels of test dataset
        batch_size (int, optional): Batch size if training discriminate model
        seeds (numpy.array, optional): Array with seeds for different runs
        test_features (torch.Tensor): Test dataset features
        input_model: Optional discriminative model object. If not given, scikit-learn
            LogisticRegression model is used
        solver (str, optional): Solver to use if using scikit-learn model

    Returns:
        dict: Dictionary of trials, active learning iterations and metrics
    """
    metric_dict = {}

    for j in range(nr_trials):
        metric_dict[j] = {}
        metric_dict[j]["Discriminative_train"] = {}
        metric_dict[j]["Discriminative_test"] = {}
        queried = []

        set_seed(seeds[j])

        if input_model is not None:
            model = input_model
            model.reset()

            train_dataset = CustomTensorDataset(X=None, Y=None)
            predict_dataset = CustomTensorDataset(X=train_features, Y=y_train)
            test_dataset = CustomTensorDataset(X=test_features, Y=y_test)
            predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        else:
            model = LogisticRegression(solver=solver)

        is_in_pool = torch.full_like(torch.Tensor(y_train), True, dtype=torch.bool)

        for i in range(al_it + 1):

            # Random sampling until we have at least one sample for each class
            if len(np.unique(y_train[queried])) < 2:
                point = random.sample(range(len(y_train)), 1)[0]
                is_in_pool[point] = False
                queried.append(point)
                metric_dict[j]["Discriminative_train"][i] = {
                    "MCC": np.nan, "Precision": np.nan, "Recall": np.nan, "Accuracy": np.nan, "F1": np.nan}
                metric_dict[j]["Discriminative_test"][i] = {
                    "MCC": np.nan, "Precision": np.nan, "Recall": np.nan, "Accuracy": np.nan, "F1": np.nan}
            # Sample based on classification boundary margin
            else:
                Y = torch.LongTensor(y_train[queried])

                if input_model is not None:
                    train_dataset.update(torch.Tensor(train_features[queried]), Y)
                    train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset, batch_size=batch_size, shuffle=True)
                    
                    train_preds = model.fit(train_loader).predict(dataloader=predict_dataloader)
                    point, is_in_pool = query_margin(train_preds, is_in_pool)
                    test_preds = model.predict(test_dataloader)
                else:
                    Y = y_train[queried]
                    train_preds = torch.Tensor(
                        model
                        .fit(train_features[queried], Y)
                        .predict_proba(train_features)
                    )
                    point, is_in_pool = query_margin(train_preds, is_in_pool)
                    test_preds = torch.Tensor(model.predict_proba(test_features))

                queried.append(point)

                metric_dict[j]["Discriminative_train"][i] = (
                    PerformanceMixin().analyze(y=y_train, preds=train_preds))
                metric_dict[j]["Discriminative_test"][i] = (
                    PerformanceMixin().analyze(y=y_test, preds=test_preds))
    return metric_dict
