import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


def plot_probs(df, probs, midpoint=0.5, add_labeled_points=None, soft_labels=True, subset=None):
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
                                marker=dict(size=8, color=df["label"], colorscale=px.colors.diverging.Geyser, colorbar=dict(title="Labels"), cmid=midpoint),
                                showlegend=False))

    if add_labeled_points is not None:
        fig.add_trace(go.Scattergl(x=df["x1"].values[add_labeled_points],
                                y=df["x2"].values[add_labeled_points],
                                mode="markers",
                                marker=dict(size=6.5, color=df["y"].values[add_labeled_points], colorscale=px.colors.diverging.Geyser, line=dict(width=1.5), opacity=1, cmid=midpoint),
                                showlegend=False))

    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                    width=700, height=700, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
    return fig


def plot_train_loss(loss_list, x_axis="Epoch", model="Label"):

    fig = go.Figure(go.Scatter(x=list(range(len(loss_list))), y=loss_list))
    fig.update_layout(xaxis_title=x_axis, yaxis_title="Loss", title_text=model + " model - Training Loss", template="plotly_white")

    return fig


class PlotMixin:
    def plot_dict(self, input_dict, categories, y_axis, label_dict=None):
        input_df = pd.DataFrame.from_dict(input_dict)
        input_df = input_df.stack().reset_index().rename(columns={"level_0": categories, "level_1": "Active Learning Iteration", 0: y_axis})
        # Filter out strings
        input_df = input_df.loc[~input_df[y_axis].apply(lambda x: isinstance(x, str))]

        if label_dict:
            input_df[categories] = input_df[categories].map(label_dict)

        grouped_df = input_df.groupby(categories)
        fig = go.Figure()
        for i, (name, group) in enumerate(grouped_df):
            fig.add_trace(go.Scatter(x=group["Active Learning Iteration"], y=group[y_axis], name=str(name), line=dict(color=np.array(px.colors.qualitative.Pastel + px.colors.qualitative.Bold)[i], shape="spline", smoothing=0.8)))
        fig.update_layout(height=700, template="plotly_white")

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

        idx_dict = {value: key for key, idx_list in self.label_model.wl_idx.items() for value in idx_list}

        label_dict = {list(range(len(idx_dict.keys())))[i]: "mu_" + str(list(range(len(idx_dict.keys())))[i]) + " = P(wl_" + str(item) + " = " + class_list[i] + ", Y = 1)" for i, item in enumerate(idx_dict.values())}

        fig = self.plot_dict(self.mu_dict, "Parameter", "Probability", label_dict)

        true_mu_df = pd.DataFrame(np.repeat(self.label_model.get_true_mu().detach().numpy()[:, 1][None, :], self.it + 1, axis=0))
        
        for idx in range(len(idx_dict.keys())):
            if true_values == "max cliques":
                true_values = self.label_model.max_clique_idx
            if idx in true_values:
                fig.add_trace(go.Scatter(x=true_mu_df.index,
                                         y=true_mu_df[idx],
                                        #  opacity=1,
                                         mode="lines",
                                         line=dict(dash="longdash", color=np.array(px.colors.qualitative.Pastel + px.colors.qualitative.Bold)[idx]),
                                         hoverinfo="none",
                                         showlegend=False))

        return fig

    def plot_probabilistic_labels(self):
        """Plot probabilistic labels"""

        fig = self.plot_dict(self.unique_prob_dict, "WL Configuration", "P(Y = 1|...)", self.confs)

        fig.add_trace(go.Scatter(x=list(range(self.it + 1)),
                                 y=np.repeat(0.5, self.it + 1),
                                 opacity=0.4,
                                 showlegend=False,
                                 hoverinfo="none",
                                 mode="lines",
                                 line=dict(color="black", dash="dash")))

        fig.update_yaxes(range=[-0.2, 1.2])

        return fig

    def plot_sampled_points(self):
        """Plot weak label configurations that have been sampled from over time"""

        conf_list = np.vectorize(self.confs.get)(self.unique_inverse[self.queried])

        fig = go.Figure(go.Scatter(x=list(range(1, self.it + 1)),
                                   y=conf_list,
                                   mode="markers",
                                   marker=dict(size=8,
                                            #    color=np.array(px.colors.qualitative.Pastel)[self.unique_inverse[self.queried]],
                                               line_width=0.2),
                                   text=self.y[self.queried],
                                   name="Sampled points"))
        fig.update_layout(height=600, template="plotly_white", xaxis_title="Active Learning Iteration", yaxis_title="WL Configuration")

        return fig

    def plot_sampled_classes(self):
        """Plot sampled ground truth labels over time"""

        fig = go.Figure()

        for i in self.label_model.y_set:
            fig.add_trace(go.Scatter(x=np.array(list(range(self.it)))[self.y[self.queried] == i] + 1,
                                     y=self.y[self.queried][self.y[self.queried] == i],
                                     mode="markers",
                                     marker_color=np.array(px.colors.diverging.Geyser)[[0,-1]][i],
                                     name="Class: " + str(i)))

        fig.update_layout(height=200, template="plotly_white", xaxis_title="Active Learning Iteration", yaxis_title="Ground truth label")

        return fig

    def plot_metrics(self, true_label_counts=True):
        """Plot metrics per iteration"""

        figures = []

        figures.append(self.plot_dict(self.metrics, "Metric", 0))
        figures[0].update_layout(title_text="Label model performance", yaxis_title="")
        if true_label_counts:
            metric_dict = self.label_model._analyze(self.label_model.predict_true_counts(), self.df["y"].values)
        else:
            metric_dict = self.label_model._analyze(self.label_model.predict_true(), self.df["y"].values)
        for i, (key, value) in enumerate(metric_dict.items()):
            figures[0].add_trace(go.Scatter(x=np.arange(0,self.it+1), y=np.repeat(value, self.it+1), line=dict(dash="longdash", color=np.array(px.colors.qualitative.Pastel)[i]), name=key + "*"))
        
        if self.final_model:
            figures.append(self.plot_dict(self.final_metrics, "Metric", 0))
            figures[1].update_layout(title_text="Discriminative model performance", yaxis_title="")

        for fig in figures:
            fig.show()
        # categories = "Metric"
        # y_axis=0

        # input_df = pd.DataFrame.from_dict(self.metrics)
        # input_df = input_df.stack().reset_index().rename(columns={"level_0": categories, "level_1": "Active Learning Iteration", 0: y_axis})
        # input_df = input_df.loc[~input_df[y_axis].apply(lambda x: isinstance(x, str))]
        # input_df["performance"] = "Label model performance"

        # input_df2 = pd.DataFrame.from_dict(self.final_metrics)
        # input_df2 = input_df2.stack().reset_index().rename(columns={"level_0": categories, "level_1": "Active Learning Iteration", 0: y_axis})
        # input_df2 = input_df2.loc[~input_df2[y_axis].apply(lambda x: isinstance(x, str))]
        # input_df2["performance"] = "Discriminative model performance"

        # joined_df = pd.concat([input_df, input_df2])

        # fig = px.line(joined_df, x="Active Learning Iteration", y=y_axis, color=categories, facet_col="performance", color_discrete_sequence=np.array(px.colors.qualitative.Pastel + px.colors.qualitative.Bold))
        # fig.update_layout(height=700, template="plotly_white", yaxis_title="")
        # fig.for_each_annotation(lambda a: a.update(text=a.text.replace("performance=", "")))

        # return fig

    def plot_iterations(self):
        """Plot sampled weak label configurations, ground truth labels and resulting probabilistic labels over time"""

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.2, 0.1, 0.7])

        fig.add_trace(self.plot_sampled_points().data[0], row=1, col=1)
        for i in range(self.label_model.y_dim):
            fig.add_trace(self.plot_sampled_classes().data[i], row=2, col=1)
        for i in range(len(self.unique_idx) + 1):
            fig.add_trace(self.plot_probabilistic_labels().data[i], row=3, col=1)
            
        fig.update_layout(height=1200, template="plotly_white")
        fig.update_xaxes(row=3, col=1, title_text="Active Learning Iteration")
        fig.update_yaxes(row=1, col=1, title_text="WL Configuration")
        fig.update_yaxes(row=2, col=1, title_text="Ground truth label")
        fig.update_yaxes(row=3, col=1, title_text="P(Y = 1|...)", range=[-0.2, 1.2])

        return fig

    def plot_animation(self):

        figures = []
        figures.append(self._plot_animation(self.prob_dict))
        if self.final_model:
            figures.append(self._plot_animation(self.final_prob_dict))

        for fig in figures:
            fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                        width=1000, height=1000, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
            fig.show()

    def _plot_animation(self, prob_dict):

        probs_df = pd.DataFrame.from_dict(prob_dict)
        probs_df = probs_df.stack().reset_index().rename(columns={"level_0": "x", "level_1": "iteration", 0: "prob_y"})
        probs_df = probs_df.merge(self.df, left_on="x", right_index=True)

        return px.scatter(probs_df, x="x1", y="x2", color="prob_y", animation_frame="iteration", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0,-1]], color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)

    def color_df(self):

        self.first_labels = np.round(self.prob_dict[0].clip(0,1))
        self.second_labels = np.round(self.prob_dict[max(self.prob_dict.keys())].clip(0,1))

        return self.df.style.apply(self._color_df, axis=None)

    def _color_df(self, df):
        c1 = 'background-color: transparent'
        c2 = 'background-color: green'
        c3 = 'background-color: red'
        df1 = pd.DataFrame(c1, index=df.index, columns=df.columns)
        idx = np.where((self.first_labels != self.df["y"]) & (self.second_labels == self.df["y"]))
        print("Wrong to correct:", len(idx[0]))
        for i in range(len(idx[0])):
            df1.loc[idx[0][i], :] = c2
        idx = np.where((self.first_labels == self.df["y"]) & (self.second_labels != self.df["y"]))
        print("Correct to wrong:", len(idx[0]))
        for i in range(len(idx[0])):
            df1.loc[idx[0][i], :] = c3
        return df1

    def color_cov(self):
        full_lm = np.append(self.label_matrix, self.df["y"].values[:, None], axis=1)
        cliques = self.label_model.cliques.copy()
        cliques.append([full_lm.shape[1]-1])
        psi_y, _ = self.label_model._get_psi(full_lm, cliques, full_lm.shape[1])
        print("Green = expected conditional independence")
        return pd.DataFrame(np.linalg.pinv(np.cov(psi_y.T))).style.apply(self._color_cov, axis=None)

    def _color_cov(self, df):
        c1 = 'background-color: red'
        c2 = 'background-color: green'
        df1 = pd.DataFrame(c1, index=df.index, columns=df.columns)
        idx = np.where(self.label_model.mask)
        for i in range(len(idx[0])):
            df1.loc[(idx[0][i], idx[1][i])] = c2
        return df1

    def plot_true_vs_predicted_posteriors(self):
        true_probs = self.label_model.predict_true()[:, 1]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["True vs predicted posteriors before active learning", "True vs predicted posteriors after active learning"])

        for i, probs in enumerate([self.prob_dict[self.it], self.prob_dict[0]]):
            fig.add_trace(go.Scatter(x=true_probs, y=probs, mode='markers', showlegend=False, marker_color=np.array(px.colors.qualitative.Pastel)[0]), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100), line=dict(dash="longdash", color=np.array(px.colors.qualitative.Pastel)[1]), showlegend=False), row=1, col=i+1)

        fig.update_yaxes(range=[0, 1], title_text="True")
        fig.update_xaxes(range=[0, 1], title_text="Predicted")
        fig.update_layout(template="plotly_white", width=1000, height=500)
        fig.show()

    

        



