import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_probs(df, probs, add_labeled_points=None, soft_labels=True, subset=None):
    """Plot data points with hard labels or estimated probability of one class"""

    if soft_labels:
        probs = probs[:, 1]

    if subset is not None:
        df = df.iloc[subset, :]
        probs = probs[subset]

    df["label"] = probs

    fig = go.Figure(go.Scattergl(x=df["x1"],
                                 y=df["x2"],
                                 mode="markers",
                                 hovertext=df["label"],
                                 hoverinfo="text",
                                 marker=dict(size=8, color=df["label"], colorscale=px.colors.diverging.Geyser, colorbar=dict(title="Labels"), cmid=0.5),
                                 showlegend=False))

    if add_labeled_points is not None:
        fig.add_trace(go.Scattergl(x=df["x1"].values[add_labeled_points],
                                   y=df["x2"].values[add_labeled_points],
                                   mode="markers",
                                   marker=dict(size=6.5, color=df["label"].values[add_labeled_points], colorscale=px.colors.diverging.Geyser, line=dict(width=1.5), opacity=1),
                                   showlegend=False))

    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      width=700, height=700, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
    return fig


def plot_accuracies(accuracies, prob_accuracy=None):
    """Plot accuracy per iteration"""

    x = list(range(len(accuracies)))

    fig = go.Figure(data=go.Scatter(x=x, y=accuracies))

    # Probabilistic label accuracy
    if prob_accuracy:
        fig.add_trace(go.Scatter(x=x, y=np.repeat(prob_accuracy, len(accuracies))))

    from IPython.display import HTML, display
    display(HTML(fig.to_html()))

