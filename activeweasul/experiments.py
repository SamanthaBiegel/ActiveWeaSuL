import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import entropy
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import torch
from tqdm import tqdm

from active_weasul import ActiveWeaSuLPipeline, set_seed
from performance import PerformanceMixin


def process_metric_dict(metric_dict, strategy_string, remove_test=False):

    metric_df = (
        pd.DataFrame(metric_dict).stack().apply(pd.Series).stack().reset_index()
        .rename(columns={"level_0": "Number of labeled points", "level_1": "Model", "level_2": "Metric", 0: "Value"})
    )

    metric_df[["Model", "Set"]] = metric_df["Model"].str.split("_", expand=True)
    metric_df["Approach"] = strategy_string

    if remove_test:
        metric_df = metric_df[metric_df["Set"] != "test"]

    return metric_df


def process_exp_dict(exp_dict, strategy_string):

    exp_df = pd.concat({i: process_metric_dict(exp_dict[i], strategy_string).drop(index=1) for i in exp_dict.keys()})

    return exp_df


def add_baseline(metric_dfs, al_it):

    baseline_df = (
        pd.concat([metric_dfs[(metric_dfs["Number of labeled points"] == 0)
        & (metric_dfs["Approach"] == "Active WeaSuL")].groupby(["Metric", "Set", "Model"])
        .mean().reset_index()]*(al_it+1), ignore_index=True)
    )

    idx = baseline_df.index
    baseline_df["Number of labeled points"] = idx[:al_it+1].repeat(len(baseline_df)/(al_it+1))
    baseline_df["Approach"] = "Weak supervision by itself"
    baseline_df["Run"] = 0
    baseline_df["Dash"] = "n"

    return pd.concat([metric_dfs, baseline_df])


def add_optimal(metric_dfs, al_it, optimal_generative_test, optimal_discriminative_test):

    optimal_lm = pd.DataFrame(optimal_generative_test, index=range(al_it+1)).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Metric", 0: "Value"})
    optimal_lm["Run"] = 0
    optimal_lm["Model"] = "Generative"
    optimal_lm["Approach"] = "Upper bound"
    optimal_lm["Dash"] = "y"
    optimal_lm["Set"] = "test"

    optimal_dm = pd.DataFrame(optimal_discriminative_test, index=range(al_it+1)).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Metric", 0: "Value"})
    optimal_dm["Run"] = 0
    optimal_dm["Model"] = "Discriminative"
    optimal_dm["Approach"] = "Upper bound"
    optimal_dm["Dash"] = "y"
    optimal_dm["Set"] = "test"

    return pd.concat([metric_dfs, optimal_lm, optimal_dm])


def plot_metrics(metric_df, filter_metrics=["Accuracy"], plot_train=False):

    if not plot_train:
        metric_df = metric_df[metric_df.Set != "train"]

    lines = list(metric_df.Approach.unique())

    colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"][:len(lines)]

    metric_df = metric_df[metric_df["Metric"].isin(filter_metrics)]

    sns.set(style="whitegrid")
    ax = sns.relplot(data=metric_df, x="Number of labeled points", y="Value", col="Model", row="Set",
                     kind="line", hue="Approach", estimator="mean", ci=68, n_boot=100, legend=False, palette=sns.color_palette(colors))

    show_handles = [ax.axes[0][0].lines[i] for i in range(len(lines))]
    show_labels = lines
    ax.axes[len(ax.axes)-1][len(ax.axes[0])-1].legend(handles=show_handles, labels=show_labels, loc="lower right")

    ax.set_ylabels("")
    ax.set_titles("{col_name}")

    # return ax


def active_weasul_experiment(nr_trials, al_it, label_matrix, y_train, cliques,
                             class_balance, query_strategy, starting_seed=76, seeds=None, 
                             discr_model_frequency=1, penalty_strength=1, batch_size=20,
                             final_model=None, train_dataset=None, test_dataset=None,
                             label_matrix_test=None, y_test=None, randomness=0):

    al_metrics = {}
    al_probs = {}
    al_queried = {}
    al_entropies = {}

    if seeds is None:
        seeds = np.random.randint(0, 1000, nr_trials)

    for i in tqdm(range(nr_trials), desc="Trials"):
        seed = seeds[i]
        # final_model.reset()
        al = ActiveWeaSuLPipeline(it=al_it,
                                  penalty_strength=penalty_strength,
                                  query_strategy=query_strategy,
                                  randomness=randomness,
                                  final_model=final_model,
                                  batch_size=batch_size,
                                  discr_model_frequency=discr_model_frequency,
                                  starting_seed=starting_seed,
                                  seed=seed)

        _ = al.run_active_weasul(label_matrix=label_matrix,
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
            bucket_list = al.unique_inverse[al.queried[:j+1]]
            al_entropies[i].append(entropy([len(np.where(bucket_list == j)[0])/len(bucket_list) for j in range(len(np.unique(al.unique_inverse)))]))

        # plot_metrics(process_metric_dict(al.metrics, query_strategy), filter_metrics=["MCC"])
        # plt.show()
        # plot_probs(df, al.probs["Generative_train"][al_it-1], soft_labels=False, add_labeled_points=al.queried[:al_it-1]).show()

    return al_metrics, al_queried, al_probs, al_entropies


def query_margin(preds, is_in_pool):
    margin = torch.abs(preds[:,1] - preds[:,0])
    minimum = torch.min(margin[is_in_pool]).item()
    chosen_points = torch.where((margin == minimum) & (is_in_pool))[0]
    point_idx = torch.randint(0,len(chosen_points), (1,1))
    point = chosen_points[point_idx].item()
    is_in_pool[point] = False
    return point, is_in_pool


def active_learning_experiment(nr_trials, al_it, model, features, y_train, y_test, batch_size, seeds, train_dataset, predict_dataloader, test_dataloader, test_features):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_dict = {}

    for j in tqdm(range(nr_trials), desc="Trials"):
        metric_dict[j] = {}
        metric_dict[j]["Discriminative_train"] = {}
        metric_dict[j]["Discriminative_test"] = {}
        queried = []

        model.reset()

        is_in_pool = torch.full_like(torch.Tensor(y_train), True, dtype=torch.bool).to(device)

        set_seed(seeds[j])

        for i in range(al_it + 1):

            if (len(queried) < 2) or (len(np.unique(y_train[queried])) < 2):
                point = random.sample(range(len(y_train)), 1)[0]
                is_in_pool[point] = False
                queried.append(point)
                metric_dict[j]["Discriminative_train"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
                metric_dict[j]["Discriminative_test"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
            else:
                Y = torch.LongTensor(y_train[queried])

                feature_subset = torch.Tensor(features[queried, :])

                train_dataset.update(feature_subset, Y)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                train_preds = model.fit(train_loader).predict(dataloader=predict_dataloader)
                query, is_in_pool = query_margin(train_preds, is_in_pool)
                queried.append(query)

                test_preds = model.predict(test_dataloader)

                metric_dict[j]["Discriminative_train"][i] = PerformanceMixin().analyze(y=y_train, preds=train_preds)
                metric_dict[j]["Discriminative_test"][i] = PerformanceMixin().analyze(y=y_test, preds=test_preds)

    return metric_dict

def synthetic_al_experiment(nr_trials, al_it, features, y_train, y_test, batch_size, seeds, train_dataset, predict_dataloader, test_dataloader, test_features):

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
                metric_dict[j]["Discriminative_train"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
                metric_dict[j]["Discriminative_test"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
            else:      
                Y = y_train[queried]
                df_1 = features.iloc[queried]
                train_preds = model.fit(df_1.loc[:, ["x1", "x2"]].values, Y).predict_proba(features.values)
                    
                queried.append(np.argmin(np.abs(train_preds[:, 1] - train_preds[:, 0])).item())

                test_preds = model.predict_proba(test_features)

                metric_dict[j]["Discriminative_train"][i] = PerformanceMixin().analyze(y=y_train, preds=torch.Tensor(train_preds))
                metric_dict[j]["Discriminative_test"][i] = PerformanceMixin().analyze(y=y_test, preds=torch.Tensor(test_preds))

    return metric_dict


def bucket_entropy_experiment(nr_trials, al_it, label_matrix, y_train, cliques, class_balance, starting_seed, seeds, query_strategy, randomness):

    entropies = {}
    for i in tqdm(range(nr_trials), desc="Repetitions"):
        seed = seeds[i]
        it = al_it

        al = ActiveWeaSuLPipeline(it=it,
                                  query_strategy=query_strategy,
                                  randomness=randomness,
                                  starting_seed=starting_seed,
                                  seed=seed)

        _ = al.run_active_weasul(label_matrix=label_matrix, y_train=y_train, cliques=cliques, class_balance=class_balance)

        entropy_sampled_buckets = []

        for j in range(it):
            bucket_list = al.unique_inverse[al.queried[:j+1]]
            entropy_sampled_buckets.append(entropy([len(np.where(bucket_list == j)[0])/len(bucket_list) for j in range(6)]))

        entropies[i] = entropy_sampled_buckets

    return entropies
