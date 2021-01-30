import torch.nn as nn

from activeweasul.discriminative_model import DiscriminativeModel
from activeweasul.performance import PerformanceMixin


class LogisticRegression(PerformanceMixin, DiscriminativeModel):
    """Logistic regression model.

    Methods for training and predicting come from DiscriminativeModel base class.
    """

    def __init__(self, input_dim, output_dim, lr, n_epochs, soft_labels=True, hide_progress_bar=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.soft_labels = soft_labels
        self.hide_progress_bar = hide_progress_bar
        self.reset()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    def reset(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim)
