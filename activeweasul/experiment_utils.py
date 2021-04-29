import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import entropy
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm_notebook as tqdm

from active_weasul import ActiveWeaSuLPipeline, set_seed
from performance import PerformanceMixin


def process_metric_dict(metric_dict, strategy_string):

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

    exp_df = pd.concat({
        i: process_metric_dict(exp_dict[i], strategy_string).drop(index=1)
        for i in exp_dict.keys()
    })
    return exp_df


def process_entropies(entropy_dict, approach_string):

    entropies_df = pd.DataFrame.from_dict(entropy_dict).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
    entropies_df["Approach"] = approach_string

    entropies_df["Number of labeled points"] = entropies_df["Number of labeled points"].apply(lambda x: x+1)
    entropies_df = entropies_df[entropies_df["Number of labeled points"] < 51]

    return entropies_df


def add_weak_supervision_baseline(metric_dfs, al_it):

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


def active_weasul_experiment(
    nr_trials, al_it, label_matrix, y_train, cliques, class_balance, query_strategy,
    starting_seed=76, seeds=None, discr_model_frequency=1, penalty_strength=1, batch_size=20,
    discriminative_model=None, train_dataset=None, test_dataset=None, label_matrix_test=None,
        y_test=None, randomness=0):

    al_metrics = {}
    al_probs = {}
    al_queried = {}
    al_entropies = {}

    if seeds is None:
        seeds = np.random.randint(0, 1000, nr_trials)

    for i in tqdm(range(nr_trials), desc="Trials"):
        seed = seeds[i]

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
        al_probs[i] = al.probs
        al_queried[i] = al.queried

        al_entropies[i] = []

        for j in range(al_it):
            bucket_list = al.unique_inverse[al.queried[:j + 1]]
            al_entropies[i].append(
                entropy([
                    len(np.where(bucket_list == j)[0]) / len(bucket_list)
                    for j in range(len(np.unique(al.unique_inverse)))
                ]))

    return al_metrics, al_entropies


def query_margin(preds, is_in_pool):
    margin = torch.abs(preds[:, 1] - preds[:, 0])
    minimum = torch.min(margin[is_in_pool]).item()
    chosen_points = torch.where((margin == minimum) & (is_in_pool))[0]
    point_idx = torch.randint(0, len(chosen_points), (1, 1))
    point = chosen_points[point_idx].item()
    is_in_pool[point] = False
    return point, is_in_pool


def active_learning_experiment(
    nr_trials, al_it, model, features, y_train, y_test, batch_size, seeds, train_dataset,
        predict_dataloader, test_dataloader, test_features):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_dict = {}

    for j in tqdm(range(nr_trials), desc="Trials"):
        metric_dict[j] = {}
        metric_dict[j]["Discriminative_train"] = {}
        metric_dict[j]["Discriminative_test"] = {}
        queried = []

        set_seed(seeds[j])

        model.reset()

        is_in_pool = torch.full_like(torch.Tensor(y_train), True, dtype=torch.bool).to(device)

        for i in range(al_it + 1):

            if len(np.unique(y_train[queried])) < 2:
                point = random.sample(range(len(y_train)), 1)[0]
                is_in_pool[point] = False
                queried.append(point)
                metric_dict[j]["Discriminative_train"][i] = {
                    "MCC": np.nan, "Precision": np.nan, "Recall": np.nan, "Accuracy": np.nan, "F1": np.nan}
                metric_dict[j]["Discriminative_test"][i] = {
                    "MCC": np.nan, "Precision": np.nan, "Recall": np.nan, "Accuracy": np.nan, "F1": np.nan}
            else:
                Y = torch.LongTensor(y_train[queried])

                feature_subset = torch.Tensor(features[queried, :])

                train_dataset.update(feature_subset, Y)
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset, batch_size=batch_size, shuffle=True)
                train_preds = model.fit(train_loader).predict(dataloader=predict_dataloader)
                point, is_in_pool = query_margin(train_preds, is_in_pool)
                queried.append(point)

                test_preds = model.predict(test_dataloader)

                metric_dict[j]["Discriminative_train"][i] = (
                    PerformanceMixin().analyze(y=y_train, preds=train_preds))
                metric_dict[j]["Discriminative_test"][i] = (
                    PerformanceMixin().analyze(y=y_test, preds=test_preds))
    return metric_dict


def synthetic_al_experiment(
    nr_trials, al_it, features, y_train, y_test, batch_size, seeds, train_dataset,
        predict_dataloader, test_dataloader, test_features):

    metric_dict = {}

    for j in tqdm(range(nr_trials), desc="Trials"):
        metric_dict[j] = {}
        metric_dict[j]["Discriminative_train"] = {}
        metric_dict[j]["Discriminative_test"] = {}
        queried = []

        model = LogisticRegression()

        set_seed(seeds[j])

        for i in range(len(queried), al_it + 1):

            if (len(queried) < 2) or (len(np.unique(y_train[queried])) < 2):
                queried.append(random.sample(range(len(y_train)), 1)[0])
                metric_dict[j]["Discriminative_train"][i] = {
                    "MCC": np.nan, "Precision": np.nan, "Recall": np.nan, "Accuracy": np.nan, "F1": np.nan}
                metric_dict[j]["Discriminative_test"][i] = {
                    "MCC": np.nan, "Precision": np.nan, "Recall": np.nan, "Accuracy": np.nan, "F1": np.nan}
            else:
                Y = y_train[queried]
                df_1 = features.iloc[queried]
                train_preds = (
                    model
                    .fit(df_1.loc[:,].values, Y)
                    .predict_proba(features.values)
                )
                queried.append(np.argmin(np.abs(train_preds[:, 1] - train_preds[:, 0])).item())
                test_preds = model.predict_proba(test_features)
                metric_dict[j]["Discriminative_train"][i] = (
                    PerformanceMixin().analyze(y=y_train, preds=torch.Tensor(train_preds)))
                metric_dict[j]["Discriminative_test"][i] = (
                    PerformanceMixin().analyze(y=y_test, preds=torch.Tensor(test_preds)))
    return metric_dict
