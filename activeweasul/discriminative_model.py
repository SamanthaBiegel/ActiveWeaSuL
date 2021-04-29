import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DiscriminativeModel(nn.Module):
    """Discriminative model base class.

    Provides training functionality to a classifier.
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.average_train_losses = []
        self.average_val_losses = []

        self.min_val_loss = 1e15

    def fit(self, train_dataloader, val_dataloader=None):
        """Train classifier"""
        self.train_dataloader = train_dataloader

        self.last_updated_min_val_loss = 0
        self.min_val_loss = 1e15

        self.train()

        if self.soft_labels:
            loss_f = lambda input, target: F.binary_cross_entropy_with_logits(
                input, target)
        else:
            loss_f = F.cross_entropy

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

        for epoch in range(self.n_epochs):

            train_losses = 0
            val_losses = 0
            train_counts = 0
            val_counts = 0

            # Train model
            for batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()

                batch_features = batch_features.to(self.device)
                batch_logits = self.forward(batch_features)
                batch_labels = batch_labels.to(self.device)

                loss = loss_f(batch_logits, batch_labels)

                loss.backward()

                optimizer.step()

                count = len(batch_labels)
                train_losses += (loss * count).item()
                train_counts += count

            self.average_train_losses.append(train_losses / train_counts)

            if self.early_stopping:

                # Validation loss
                for batch_features, batch_labels in val_dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_logits = self.forward(batch_features)
                    batch_labels = batch_labels.to(self.device)

                    loss = loss_f(batch_logits, batch_labels)

                    count = len(batch_labels)
                    val_losses += (loss * count).item()
                    val_counts += count

                current_val_loss = val_losses / val_counts
                self.average_val_losses.append(current_val_loss)

                # Early stopping
                if current_val_loss < self.min_val_loss:
                    self.min_val_loss = current_val_loss
                    self.last_updated_min_val_loss = 0
                    torch.save(self.state_dict(), self.checkpoint)
                else:
                    self.last_updated_min_val_loss += 1
                    if self.last_updated_min_val_loss >= self.patience:
                        self.load_state_dict(torch.load(self.checkpoint))
                        return self
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
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(self.device)
            batch_logits = self.forward(batch_features)
            preds.extend(F.softmax(batch_logits, dim=1))
        preds = torch.stack(preds)

        if assign_train_preds:
            self.preds = preds
        return preds
