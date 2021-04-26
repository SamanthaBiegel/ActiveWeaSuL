import torch.nn as nn

from activeweasul.discriminative_model import DiscriminativeModel
from activeweasul.performance import PerformanceMixin


class LogisticRegression(PerformanceMixin, DiscriminativeModel):
    """Logistic regression model.

    Methods for training and predicting come from DiscriminativeModel base class.
    """

    def __init__(
        self, input_dim, output_dim, lr, n_epochs, early_stopping=False,
            checkpoint="../checkpoints/LG_checkpoint.pt", patience=20, soft_labels=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.patience = patience
        self.soft_labels = soft_labels

        self.reset()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def reset(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim).to(self.device)
