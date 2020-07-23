import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def cross_entropy_soft_labels(predictions, targets):
    """Implement cross entropy loss for probabilistic labels"""

    y_dim = targets.shape[1]
    loss = torch.zeros(predictions.shape[0])
    for y in range(y_dim):
        loss_y = F.cross_entropy(predictions, predictions.new_full((predictions.shape[0],), y, dtype=torch.long),
                                 reduction="none")
        loss += targets[:, y] * loss_y

    return loss.mean()


def fit_predict_fm(train_set, labels, input_dim, output_dim, lr, batch_size, n_epochs, soft_labels=True, subset=None):
    """Fit final logistic regression model on probabilistic or hard labels and predict"""

    # writer = SummaryWriter()

    if soft_labels:
        target = torch.Tensor(labels)
        loss_f = cross_entropy_soft_labels
    else:
        target = torch.LongTensor(labels)
        loss_f = F.cross_entropy

    train = torch.Tensor(train_set)

    if subset is not None:
        train_tensor = torch.utils.data.TensorDataset(train[subset], target[subset])
    else:
        train_tensor = torch.utils.data.TensorDataset(train, target)

    train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)

    # if model is None:
    model = LogisticRegression(input_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(n_epochs):
        for i, (batch_features, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_logits = model(batch_features)
            loss = loss_f(batch_logits, batch_labels)
            # writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

    model.eval()
    logits = model(train)

    preds = F.softmax(logits, dim=1).detach().numpy()

    # writer.flush()
    # writer.close()

    return preds
