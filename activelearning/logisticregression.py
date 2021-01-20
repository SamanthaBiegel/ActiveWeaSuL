import torch.nn as nn

from discriminative_model import DiscriminativeModel
from performance import PerformanceMixin


class LogisticRegression(PerformanceMixin, DiscriminativeModel):
    """Logistic regression model.

    Methods for training and predicting come from DiscriminativeModel base class.
    """

    def __init__(self, input_dim, output_dim, lr, n_epochs, soft_labels=True):
        super().__init__()
        self.lr = lr
        self.n_epochs = n_epochs
        self.soft_labels = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reset()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    def reset(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim)
