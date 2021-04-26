import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_probs(df, probs, midpoint=0.5, add_labeled_points=None, soft_labels=True, subset=None):
    """Plot data points with hard labels or estimated probability of one class"""

    if soft_labels:
        probs = probs[:, 1]
    if subset is not None:
        df = df.iloc[subset, :]
        probs = probs[subset]
    df["label"] = probs

    fig = go.Figure(
        go.Scattergl(
            x=df["x1"], y=df["x2"], mode="markers", hovertext=df["label"], hoverinfo="text",
            marker=dict(
                size=6, color=df["label"], colorscale=px.colors.diverging.Geyser,
                colorbar=dict(title="Labels"), cmid=midpoint),
            showlegend=False))
    if add_labeled_points is not None:
        fig.add_trace(
            go.Scattergl(
                x=df["x1"].values[add_labeled_points], y=df["x2"].values[add_labeled_points],
                mode="markers", marker=dict(
                    size=6, color=df["y"].values[add_labeled_points],
                    colorscale=px.colors.diverging.Geyser, line=dict(width=1.5),
                    opacity=1, cmid=midpoint),
                showlegend=False))
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1), width=700, height=700,
        xaxis_title="x1", yaxis_title="x2", template="plotly_white")
    return fig


def plot_train_loss(loss_list, x_axis="Epoch"):

    fig = go.Figure(go.Scatter(x=list(range(len(loss_list))), y=loss_list))
    fig.update_layout(
        xaxis_title=x_axis, yaxis_title="Loss", title_text="Label Model Loss",
        template="plotly_white")
    return fig


class PlotMixin:
    def plot_dict(self, input_dict, categories, y_axis, label_dict=None):
        input_df = pd.DataFrame.from_dict(input_dict)
        input_df = (
            input_df
            .stack()
            .reset_index()
            .rename(columns={
                "level_0": categories,
                "level_1": "Active Learning Iteration", 0: y_axis
            })
        )
        # Filter out strings
        input_df = input_df.loc[~input_df[y_axis].apply(lambda x: isinstance(x, str))]

        if label_dict:
            input_df[categories] = input_df[categories].map(label_dict)

        grouped_df = input_df.groupby(categories, sort=False)
        fig = go.Figure()
        for i, (name, group) in enumerate(grouped_df):
            fig.add_trace(go.Scatter(
                x=group["Active Learning Iteration"], y=group[y_axis], name=str(name),
                line=dict(
                    color=np.array(px.colors.qualitative.Pastel + px.colors.qualitative.Bold)[i],
                    shape="spline", smoothing=0.8)))
        fig.update_layout(
            height=700, template="plotly_white", xaxis_title="Active Learning Iteration")
        return fig

    def plot_parameters(self, true_values="max cliques"):
        """Plot each parameter against number of iterations"""

        class_list = []
        for key, value in self.label_model.wl_idx.items():
            if len(value) == 2:
                class_list.extend(["0", "1"])
            if len(value) == 4:
                class_list.extend(["00", "10", "01", "11"])
            if len(value) == 8:
                class_list.extend(["000", "100", "010", "110", "001", "101", "011", "111"])

        idx_dict = {
            value: key
            for key, idx_list in self.label_model.wl_idx.items() for value in idx_list
        }
        label_dict = {
            list(range(len(idx_dict.keys())))[i]:
            "mu_"
            + str(list(range(len(idx_dict.keys())))[i])
            + " = P(wl_" + str(item)
            + " = "
            + class_list[i]
            + ", Y = 1)"
            for i, item in enumerate(idx_dict.values())
        }

        fig = self.plot_dict(self.mu_dict, "Parameter", "Probability", label_dict)

        true_mu_df = pd.DataFrame(
            np.repeat(
                self.label_model.get_true_mu(self.y_train).detach().numpy()[:, 1][None, :],
                self.it + 1, axis=0))

        for idx in range(len(idx_dict.keys())):
            if true_values == "max cliques":
                true_values = self.label_model.max_clique_idx
            if idx in true_values:
                fig.add_trace(go.Scatter(
                    x=true_mu_df.index, y=true_mu_df[idx], mode="lines", line=dict(
                        dash="longdash",
                        color=np.array(px.colors.qualitative.Pastel
                                       + px.colors.qualitative.Bold)[idx]),
                    hoverinfo="none", showlegend=False))
        return fig

    def plot_probabilistic_labels(self):
        """Plot probabilistic labels"""

        fig = self.plot_dict(self.probs["bucket_labels_train"], "Bucket", "P(Y = 1|...)", self.bucket_conf_dict)
        true_posteriors = self.label_model.predict_true(self.y_train)[self.unique_idx, 1]
        for i in range(len(self.unique_idx)):
            fig.add_trace(go.Scatter(
                x=list(range(self.it + 1)), y=np.repeat(true_posteriors[i], self.it + 1),
                name=self.bucket_conf_dict[i] + "*", line=dict(
                    dash="longdash",
                    color=np.array(px.colors.qualitative.Pastel + px.colors.qualitative.Bold)[i]),
                hoverinfo="none"))
        fig.update_yaxes(range=[- 0.2, 1.2])
        return fig

    def plot_sampled_buckets(self):
        """Plot sampled buckets for every iteration"""

        conf_list = np.vectorize(self.bucket_conf_dict.get)(self.unique_inverse[self.queried])
        fig = go.Figure(go.Scatter(
            x=list(range(1, self.it + 1)), y=conf_list, mode="markers", marker=dict(
                size=8, line_width=0.2), text=self.y_train[self.queried], name="Sampled points"))
        fig.update_layout(
            height=600, template="plotly_white", xaxis_title="Active Learning Iteration",
            yaxis_title="Bucket")
        return fig

    def plot_sampled_classes(self):
        """Plot sampled ground truth labels for every iteration"""
        fig = go.Figure()
        for i in self.label_model.y_set:
            fig.add_trace(go.Scatter(
                x=np.array(list(range(self.it)))[self.y_train[self.queried] == i] + 1,
                y=self.y_train[self.queried][self.y_train[self.queried] == i],
                mode="markers",
                marker_color=np.array(px.colors.diverging.Geyser)[[0, -1]][i],
                name="Class: " + str(i)))
        fig.update_layout(
            height=200, template="plotly_white", xaxis_title="Active Learning Iteration",
            yaxis_title="Ground truth label")
        return fig

    def plot_metrics(self, plot_test=True):
        """Plot metrics per iteration"""

        figures = []
        figures.append(self.plot_dict(self.metrics["Generative_train"], "Metric", 0))
        figures[0].update_layout(title_text="Generative model train performance", yaxis_title="")
        metric_dict = self.label_model.analyze(self.y_train,
            self.label_model.predict_true(self.y_train))
        for i, (key, value) in enumerate(metric_dict.items()):
            figures[0].add_trace(go.Scatter(
                x=np.arange(0, self.it + 1), y=np.repeat(value, self.it + 1), line=dict(
                    dash="longdash", color=np.array(px.colors.qualitative.Pastel)[i]),
                name=key + "*"))

        if plot_test:
            figures.append(self.plot_dict(self.metrics["Generative_test"], "Metric", 0))
            figures[-1].update_layout(title_text="Generative model test performance", yaxis_title="")
            metric_dict = self.label_model.analyze(self.y_test,
                self.label_model.predict_true(self.y_train, self.y_test, self.label_matrix_test))
            for i, (key, value) in enumerate(metric_dict.items()):
                figures[-1].add_trace(go.Scatter(
                    x=np.arange(0, self.it + 1), y=np.repeat(value, self.it + 1), line=dict(
                        dash="longdash", color=np.array(px.colors.qualitative.Pastel)[i]),
                    name=key + "*"))

        if self.final_model:
            figures.append(self.plot_dict(self.metrics["Discriminative_train"], "Metric", 0))
            figures[-1].update_layout(title_text="Discriminative model train performance", yaxis_title="")
            if plot_test:
                figures.append(self.plot_dict(self.metrics["Discriminative_test"], "Metric", 0))
                figures[-1].update_layout(title_text="Discriminative model test performance", yaxis_title="")

        for fig in figures:
            fig.show()

    def plot_iterations(self):
        """Plot sampled buckets ground truth labels and resulting
        probabilistic labels for every iteration"""

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.2, 0.1, 0.7])

        fig.add_trace(self.plot_sampled_buckets().data[0], row=1, col=1)
        for i in range(self.label_model.y_dim):
            fig.add_trace(self.plot_sampled_classes().data[i], row=2, col=1)
        for i in range(len(self.unique_idx)*2):
            fig.add_trace(self.plot_probabilistic_labels().data[i], row=3, col=1)

        fig.update_layout(height=1200, template="plotly_white")
        fig.update_xaxes(row=3, col=1, title_text="Active Learning Iteration")
        fig.update_yaxes(row=1, col=1, title_text="Bucket")
        fig.update_yaxes(row=2, col=1, title_text="Ground truth label")
        fig.update_yaxes(row=3, col=1, title_text="P(Y = 1|...)", range=[-0.2, 1.2])

        return fig

    def plot_animation(self, X):

        figures = []
        figures.append(self._plot_animation(self.probs["Generative_train"], X))
        if self.final_model:
            figures.append(self._plot_animation(self.probs["Discriminative_train"], X))

        for fig in figures:
            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=1), width=1000, height=1000,
                xaxis_title="x1", yaxis_title="x2", template="plotly_white")
            fig.show()

    def _plot_animation(self, prob_dict, X):

        probs_df = pd.DataFrame.from_dict(prob_dict)
        probs_df = probs_df.stack().reset_index().rename(columns={
            "level_0": "x",
            "level_1": "iteration",
            0: "prob_y"
        })
        probs_df = probs_df.merge(X, left_on="x", right_index=True)

        return px.scatter(
            probs_df, x="x1", y="x2", color="prob_y", animation_frame="iteration",
            color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0, -1]],
            color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)

    def color_df(self, df):

        self.first_labels = np.round(self.probs["Generative_train"][0].clip(0, 1)).astype(int)
        self.second_labels = np.round(self.probs["Generative_train"][max(self.probs["Generative_train"].keys())].clip(0, 1)).astype(int)

        return df.style.apply(self._color_df, axis=None)

    def _color_df(self, df):
        c1 = 'background-color: transparent'
        c2 = 'background-color: green'
        c3 = 'background-color: red'
        df1 = pd.DataFrame(c1, index=df.index, columns=df.columns)
        idx = np.where(
            (
                self.first_labels != self.y_train
            ) & self.second_labels == self.y_train
        )
        print("Wrong to correct:", len(idx[0]))
        for i in range(len(idx[0])):
            df1.loc[idx[0][i], :] = c2
        idx = np.where(
            (
                self.first_labels == self.y_train
            ) & (
                self.second_labels != self.y_train
            )
        )
        print("Correct to wrong:", len(idx[0]))
        for i in range(len(idx[0])):
            df1.loc[idx[0][i], :] = c3
        return df1

    def color_cov(self):
        psi, _ = self.label_model.get_psi()
        psi_y = np.append(psi, (self.y_train[:, None] == [0, 1])*1, axis=1)
        print("Green = expected conditional independence")
        return pd.DataFrame(
            np.linalg.pinv(np.cov(psi_y.T))
        ).style.apply(self._color_cov, axis=None)

    def _color_cov(self, df):
        c1 = 'background-color: red'
        c2 = 'background-color: green'
        df1 = pd.DataFrame(c1, index=df.index, columns=df.columns)
        idx = np.where(self.label_model.mask)
        for i in range(len(idx[0])):
            df1.loc[(idx[0][i], idx[1][i])] = c2
        return df1

    def plot_true_vs_predicted_posteriors(self):
        true_probs = self.label_model.predict_true(self.y_train)[:, 1]
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=[
                "True vs predicted posteriors before active learning",
                "True vs predicted posteriors after active learning"
            ]
        )
        for i, probs in enumerate([self.probs["Generative_train"][self.it], self.probs["Generative_train"][0]]):
            fig.add_trace(go.Scatter(
                x=true_probs, y=probs, mode='markers', showlegend=False,
                marker_color=np.array(px.colors.qualitative.Pastel)[0]), row=1, col=i+1)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100), line=dict(
                    dash="longdash", color=np.array(px.colors.qualitative.Pastel)[1]),
                showlegend=False), row=1, col=i+1)
        fig.update_yaxes(range=[0, 1], title_text="True")
        fig.update_xaxes(range=[0, 1], title_text="Predicted")
        fig.update_layout(template="plotly_white", width=1000, height=500)
        fig.show()
