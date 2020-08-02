import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_probs(df, probs, soft_labels=True, subset=None):
    """Plot data points with hard labels or estimated probability of one class"""

    if soft_labels:
        probs = probs[:, 1]

    if subset is not None:
        df = df.iloc[subset, :]
        probs = probs[subset]

    df["label"] = probs.detach().numpy()

    fig = px.scatter(df, x="x1", y="x2", color="label", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0,-1]], color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      width=700, height=700, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
    fig.show()


def plot_accuracies(accuracies, prob_accuracy=None):
    """Plot accuracy per iteration"""

    x = list(range(len(accuracies)))

    fig = go.Figure(data=go.Scatter(x=x, y=accuracies))

    # Probabilistic label accuracy
    if prob_accuracy:
        fig.add_trace(go.Scatter(x=x, y=np.repeat(prob_accuracy, len(accuracies))))

    from IPython.display import HTML, display
    display(HTML(fig.to_html()))
