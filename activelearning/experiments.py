import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pipeline import ActiveWeaSuLPipeline


def process_metric_dict(metrics):

    metric_df = (
        pd.DataFrame(metrics).stack().apply(pd.Series).stack().reset_index()
        .rename(columns={"level_0": "Number of labeled points", "level_1": "Model", "level_2": "Metric", 0: "Value"})
    )

    metric_df[["Model", "Set"]] = metric_df["Model"].str.split("_", expand=True)

    return metric_df


def plot_metrics(metric_df):

    colors = ["#368f8b", "#ec7357"]

    sns.set(style="whitegrid")
    ax = sns.relplot(data=metric_df, x="Number of labeled points", y="Value", col="Metric", row="Model",
                    kind="line", hue="Set",legend=False, palette=sns.color_palette(colors))

    show_handles = [ax.axes[0][0].lines[0], ax.axes[0][0].lines[1]]
    show_labels = ["train", "test"]
    ax.axes[len(ax.axes)-1][len(ax.axes[0])-1].legend(handles=show_handles, labels=show_labels, loc="lower right")

    ax.set_ylabels("")
    ax.set_titles("{col_name}")
    
    plt.show()


def active_weasul_experiment(al_it, nr_trials, label_matrix, y_true, cliques,
                             class_balance, label_matrix_test, y_test, query_strategy, randomness, final_model):

    al_metrics = {}
    al_metrics["lm_metrics"] = {}
    al_metrics["fm_metrics"] = {}

    for i in tqdm(range(nr_trials), desc="Trials"):
        al = ActiveLearningPipeline(it=al_it,
                                    final_model=final_model,
                                    y_true=y_true,
                                    query_strategy=query_strategy,
                                    randomness=randomness)

        _ = al.run_active_learning(label_matrix=label_matrix,
                                   cliques=cliques,
                                   class_balance=class_balance,
                                   label_matrix_test=label_matrix_test,
                                   y_test=y_test)

        al_metrics["lm_metrics"][i] = al.metrics
        al_metrics["fm_metrics"][i] = al.final_metrics

    return al_metrics
