import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from performance import PerformanceMixin


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class DiscriminativeModel(PerformanceMixin, LogisticRegression):
    def __init__(self, df, input_dim, output_dim, lr, batch_size, n_epochs, soft_labels=True, subset=None):
        self.model_name = "Discriminative Model"
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.soft_labels = soft_labels
        self.subset = subset
        self.df = df

        super().__init__(input_dim=input_dim, output_dim=output_dim)

    def cross_entropy_soft_labels(self, predictions, targets):
        """Implement cross entropy loss for probabilistic labels"""

        y_dim = targets.shape[1]
        loss = torch.zeros(predictions.shape[0])
        for y in range(y_dim):
            loss_y = F.cross_entropy(predictions, predictions.new_full((predictions.shape[0],), y, dtype=torch.long),
                                     reduction="none")
            loss += targets[:, y] * loss_y

        return loss.mean()

    def fit(self, features, labels):
        """Fit logistic regression model on probabilistic or hard labels"""

        self.train()

        # writer = SummaryWriter()

        if self.soft_labels:
            target = torch.Tensor(labels)
            loss_f = self.cross_entropy_soft_labels
        else:
            target = torch.LongTensor(labels)
            loss_f = F.cross_entropy

        self.train_set = torch.Tensor(features)

        if self.subset is not None:
            train_tensor_set = torch.utils.data.TensorDataset(self.train_set[self.subset], target[self.subset])
        else:
            train_tensor_set = torch.utils.data.TensorDataset(self.train_set, target)

        train_loader = torch.utils.data.DataLoader(dataset=train_tensor_set, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            for i, (batch_features, batch_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_logits = self.forward(batch_features)
                loss = loss_f(batch_logits, batch_labels)    

                loss.backward()
                optimizer.step()

                # logits = self.forward(self.train)
                # preds = F.softmax(logits, dim=1).detach().numpy()
                # writer.add_scalar('final model loss', loss, epoch)
                # writer.add_scalar('final model accuracy', _accuracy(preds, y), epoch)

        # writer.flush()
        # writer.close()

        return self

    def predict(self):
        """Predict on the train set"""

        self.preds = self._predict(self.train_set)

        return self.preds

    @torch.no_grad()
    def _predict(self, input):
        """Predict on input"""

        self.eval()
        logits = self.forward(input)
        preds = F.softmax(logits, dim=1)

        return preds
