import torch
from torch.utils.data import TensorDataset


class CustomTensorDataset(TensorDataset):
    """Custom Tensor Dataset for easy updating"""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def update(self, X, Y):
        """Update dataset content

        Args:
            X (torch.Tensor): Tensor with features (columns)
            Y (torch.Tensor): Tensor with labels
        """
        self.X = X
        self.Y = Y