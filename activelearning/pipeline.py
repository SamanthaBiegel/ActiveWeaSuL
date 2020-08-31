import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm_notebook as tqdm

from label_model import LabelModel


class ActiveLearningPipeline(LabelModel):
    def __init__(self,
                 final_model_kwargs,
                 df,
                 it: int = 100,
                 n_epochs: int = 200,
                 lr: float = 1e-1,
                 active_learning: str = "probs",
                 query_strategy: str = "margin",
                 alpha: float = 0.01,
                 beta: float = 0.1,
                 add_neighbors: int = 0,
                 add_cliques: bool = True,
                 add_prob_loss: bool = False,
                 randomness: float = 0):

        self.it = it
        self.query_strategy = query_strategy
        self.alpha = alpha
        self.beta = beta
        self.add_neighbors = add_neighbors
        self.randomness = randomness
        super().__init__(final_model_kwargs=final_model_kwargs,
                         df=df,
                         n_epochs=n_epochs,
                         lr=lr,
                         active_learning=active_learning,
                         add_cliques=add_cliques,
                         add_prob_loss=add_prob_loss,
                         hide_progress_bar=True)

    def entropy(self, probs):
    
        prod = probs * torch.log(probs)
        prod[torch.isnan(prod)] = 0
        
        return - prod.sum(axis=1)

    def list_options(self, values, criterium):
        """Return options with equal informativeness according to criterium"""

        return [j for j, v in enumerate(values) if v == criterium and self.ground_truth_labels[j] == -1]

    def margin(self, probs):
        """P(Y=1|...) - P(Y=0|...)"""

        abs_diff = torch.abs(probs[:, 1] - probs[:, 0])

        return abs_diff

    def margin_strategy(self, probs):
        """List query options based on minimum margin strategy"""

        abs_diff = self.margin(probs)

        minimum = min(j for i, j in enumerate(abs_diff) if self.ground_truth_labels[i] == -1)
        
        return self.list_options(abs_diff, minimum)

    def uncertainty(self, probs):
        """1 - P(Y=1|...)"""

        return 1 - torch.max(probs, dim=1, keepdim=True).values

    def entropy_strategy(self, probs):
        """List query options based on maximum entropy strategy"""

        H = self.entropy(probs)

        maximum = max(j for i, j in enumerate(H) if self.ground_truth_labels[i] == -1)

        return self.list_options(H, maximum)

    def information_density(self):
        """Compute information density of each point"""

        I = torch.Tensor(1 / (self.N - len(self.queried)) * squareform(1 / pdist(self.X, metric="euclidean")).sum(axis=1))

        return I

    def margin_density(self, probs):
        """Query data point based on combined margin and information density measures"""

        measure = (1/self.margin(probs))**self.beta * self.information_density()**(1 - self.beta)

        maximum = max([j for i, j in enumerate(measure) if self.ground_truth_labels[i] == -1])

        return self.list_options(measure, maximum)[0]

    def query(self, probs):
        """Choose data point to label from label predictions"""

        pick = np.random.uniform()

        if pick < self.randomness:
            indices = [i for i in range(self.N) if self.ground_truth_labels[i] == -1]

        elif self.query_strategy == "margin":
            indices = self.margin_strategy(probs)

        elif self.query_strategy == "entropy":
            indices = self.entropy_strategy(probs)
        
        elif self.query_strategy == "margin_density":
            return self.margin_density(probs)

        # Make really random
        random.seed(random.SystemRandom().random())

        # Pick a random point from least confident data points
        if self.add_neighbors:
            return random.sample(indices, self.add_neighbors)
        else:
            return random.choice(indices)

    def logging(self, count, probs, selected_point=None):
        """Keep track of accuracy and other metrics"""

        if count == 0:
            self.accuracies = []
            self.queried = []
            self.prob_dict = {}
            self.unique_prob_dict = {}
            self.mu_dict = {}

        self.accuracies.append(self._accuracy(probs, self.y))
        self.prob_dict[count] = probs[:, 1].clone().detach().numpy()
        self.unique_prob_dict[count] = self.prob_dict[count][self.unique_idx]
        self.mu_dict[count] = self.mu.clone().detach().numpy().squeeze()

        if selected_point is not None:
            self.queried.append(selected_point)

        return self

    def update_parameters(self, n_queried, alpha):
        """Use update rule to adjust parameters based on sampled data points"""

        if n_queried == 1:
            self.mu_0 = self.mu.clone()

        psi, _ = self._get_psi(self.label_matrix[self.ground_truth_labels != -1])
        mu_samples = torch.Tensor(psi[self.y[self.ground_truth_labels != -1] == 1].sum(axis=0) / n_queried)[:, None]
        self.mu = self.mu_0*np.exp(-self.alpha*n_queried) + mu_samples*(1 - np.exp(-self.alpha*n_queried))

        return self.predict()

    def refine_probabilities(self, label_matrix, cliques, class_balance):
        """Iteratively refine label matrix and training set predictions with active learning strategy"""

        self.label_matrix = label_matrix
        self.ground_truth_labels = np.full_like(self.df["y"].values, -1)
        self.X = self.df[["x1", "x2"]].values
        self.y = self.df["y"].values
        nr_wl = label_matrix.shape[1]

        if self.active_learning == "cov":
            # self.add_cliques = False
            self.label_matrix = np.concatenate([self.label_matrix, self.ground_truth_labels[:, None]], axis=1)

        if self.add_neighbors:
            neigh = NearestNeighbors(n_neighbors=self.add_neighbors)
            neigh.fit(self.X)

        old_probs = self.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

        _, self.unique_idx, self.unique_inverse = np.unique(old_probs.clone().detach().numpy()[:, 1], return_index=True, return_inverse=True)

        self.logging(count=0, probs=old_probs)

        for i in tqdm(range(self.it)):
            sel_idx = self.query(old_probs)
            self.ground_truth_labels[sel_idx] = self.y[sel_idx]

            if self.active_learning == "cov":
                if self.add_neighbors:
                    neighbors_sel_idx = neigh.kneighbors(self.X[sel_idx, :][None, :], return_distance=False)
                    self.label_matrix[neighbors_sel_idx, nr_wl] = self.y[sel_idx]
                else:
                    self.label_matrix[sel_idx, nr_wl] = self.y[sel_idx]

            if self.active_learning == "update_params":
                new_probs = self.update_parameters(n_queried=i+1, alpha=self.alpha)
            else:
                # Fit label model on refined label matrix
                new_probs = self.fit(label_matrix=self.label_matrix, cliques=cliques, class_balance=class_balance, ground_truth_labels=self.ground_truth_labels).predict()

            self.logging(count=i+1, probs=new_probs, selected_point=sel_idx)

            old_probs = new_probs.clone()

            # if self.active_learning == "cov":
            #     if i == self.it - 1:
            #         break

            #     self.label_matrix = np.concatenate([self.label_matrix, np.full_like(self.df["y"].values, -1)[:,None]], axis=1)
            #     cliques.append([nr_wl + i + 1])

        self.confs = {range(len(self.unique_idx))[i]: "-".join([str(e) for e in row]) for i, row in enumerate(self.label_matrix[self.unique_idx, :])}

        return new_probs

    def plot_dict(self, input_dict, categories, y_axis, label_dict):

        input_df = pd.DataFrame.from_dict(input_dict)
        input_df = input_df.stack().reset_index().rename(columns={"level_0": categories, "level_1": "Active Learning Iteration", 0: y_axis})

        input_df[categories] = input_df[categories].map(label_dict)

        fig = px.line(input_df, x="Active Learning Iteration", y=y_axis, color=categories, color_discrete_sequence=np.array(px.colors.qualitative.Pastel + px.colors.qualitative.Bold))
        fig.update_layout(height=700, template="plotly_white")

        return fig

    def plot_parameters(self):
        """Plot each parameter against number of iterations"""

        class_list = []
        for key, value in self.wl_idx.items():
            if len(value) == 2:
                class_list.extend(["0", "1"])
            if len(value) == 4:
                class_list.extend(["00", "10", "01", "11"])
            if len(value) == 8:
                class_list.extend(["000", "100", "010", "110", "001", "101", "011", "111"])

        idx_dict = {value: key for key, idx_list in self.wl_idx.items() for value in idx_list}

        label_dict = {list(range(len(idx_dict.keys())))[i]: "mu_" + str(list(range(len(idx_dict.keys())))[i]) + " = P(wl_" + str(item) + " = " + class_list[i] + ", Y = 1)" for i, item in enumerate(idx_dict.values())}

        fig = self.plot_dict(self.mu_dict, "Parameter", "Probability", label_dict)

        true_mu_df = pd.DataFrame(np.repeat(self.get_true_mu().detach().numpy()[:, 1][None, :], self.it + 1, axis=0))

        for i in range(len(idx_dict.keys())):
            fig.add_trace(go.Scatter(x=true_mu_df.index,
                                     y=true_mu_df[i],
                                     opacity=0.4,
                                     mode="lines",
                                     line=dict(dash="dash"),
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

        fig = go.Figure(go.Scatter(x=list(range(self.it)),
                                   y=conf_list,
                                   mode="markers",
                                   marker=dict(size=8,
                                               color=np.array(px.colors.qualitative.Pastel)[self.unique_inverse[self.queried]],
                                               line_width=0.2),
                                   text=self.y[self.queried],
                                   name="Sampled points"))
        fig.update_layout(height=600, template="plotly_white", xaxis_title="Active Learning Iteration", yaxis_title="WL Configuration")

        return fig

    def plot_sampled_classes(self):
        """Plot sampled ground truth labels over time"""

        fig = go.Figure()

        for i in self.y_set:
            fig.add_trace(go.Scatter(x=np.array(list(range(self.it)))[self.y[self.queried] == i],
                                     y=self.y[self.queried][self.y[self.queried] == i],
                                     mode="markers",
                                     marker_color=np.array(px.colors.diverging.Geyser)[[0,-1]][i],
                                     name="Class: " + str(i)))

        fig.update_layout(height=200, template="plotly_white", xaxis_title="Active Learning Iteration", yaxis_title="Ground truth label")

        return fig

    def plot_iterations(self):
        """Plot sampled weak label configurations, ground truth labels and resulting probabilistic labels over time"""

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.2, 0.1, 0.7])

        fig.add_trace(self.plot_sampled_points().data[0], row=1, col=1)
        for i in range(self.y_dim):
            fig.add_trace(self.plot_sampled_classes().data[i], row=2, col=1)
        for i in range(len(self.unique_idx) + 1):
            fig.add_trace(self.plot_probabilistic_labels().data[i], row=3, col=1)
            
        fig.update_layout(height=1200, template="plotly_white")
        fig.update_xaxes(row=3, col=1, title_text="Active Learning Iteration")
        fig.update_yaxes(row=1, col=1, title_text="WL Configuration")
        fig.update_yaxes(row=2, col=1, title_text="Ground truth label")
        fig.update_yaxes(row=3, col=1, title_text="P(Y = 1|...)", range=[-0.2, 1.2])

        return fig


