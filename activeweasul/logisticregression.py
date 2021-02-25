import torch.nn as nn

from discriminative_model import DiscriminativeModel
from performance import PerformanceMixin


class LogisticRegression(PerformanceMixin, DiscriminativeModel):
    """Logistic regression model.

    Methods for training and predicting come from DiscriminativeModel base class.
    """

    def __init__(self, input_dim, output_dim, lr, n_epochs, early_stopping=False, warm_start=False, patience=20, soft_labels=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.warm_start = warm_start
        self.patience = patience
        self.soft_labels = soft_labels

        self.checkpoint = "../checkpoints/LG_checkpoint.pt"

        self.reset()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    def reset(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim).to(self.device)
