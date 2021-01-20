import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm


class DiscriminativeModel(nn.Module):
    """Discriminative model base class.

    Provides training functionality to a classifier.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cross_entropy_soft_labels(self, predictions, targets):
        """Cross entropy loss for probabilistic labels"""

        y_dim = targets.shape[1]
        loss = torch.zeros(predictions.shape[0]).to(self.device)
        for y in range(y_dim):
            loss_y = F.cross_entropy(predictions, predictions.new_full((predictions.shape[0],), y, dtype=torch.long),
                                     reduction="none")
            loss += targets[:, y] * loss_y

        return loss.mean()

    def fit(self, train_dataloader):
        """Train classifier"""

        self.train_dataloader = train_dataloader

        self.train()

        self.losses = []
        self.counts = 0
        self.average_losses = []

        if self.soft_labels:
            loss_f = self.cross_entropy_soft_labels
        else:
            loss_f = F.cross_entropy

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.n_epochs), desc="Discriminative Model Epochs"):
            for i, (batch_features, batch_labels) in enumerate(train_dataloader):
                optimizer.zero_grad()

                batch_logits = self.forward(batch_features)

                loss = loss_f(batch_logits, batch_labels)

                loss.backward()

                optimizer.step()

                count = len(batch_labels)
                self.losses.append(loss * count)
                self.counts += count
                self.average_losses.append((sum(self.losses) / self.counts).item())

        return self

    @torch.no_grad()
    def predict(self, dataloader=None, assign_train_preds=False):
        """Predict on dataset"""

        if dataloader is None:
            # Predict on train set if no dataloader provided
            dataloader = DataLoader(dataset=self.train_dataloader.dataset,
                                    shuffle=False,
                                    batch_size=self.train_dataloader.batch_size)
            assign_train_preds = True

        self.eval()

        preds = []

        for batch_features, batch_labels in dataloader:
            batch_logits = self.forward(batch_features)
            preds.extend(F.softmax(batch_logits, dim=1))

        preds = torch.stack(preds)

        if assign_train_preds:
            self.preds = preds

        return preds
