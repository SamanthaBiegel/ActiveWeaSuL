import csv
import numpy as np
import pandas as pd
from PIL import Image
import torch

from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms

from performance import PerformanceMixin
from discriminative_model import DiscriminativeModel


class VisualRelationDataset(Dataset):
    """Visual Relation Dataset"""

    def __init__(
        self, image_dir: str, df: pd.DataFrame, Y: torch.Tensor, image_size=224,
    ) -> None:
        self.image_dir = image_dir
        self.X = df.loc[:, [
            "source_img", "object_bbox", "subject_bbox", "object_category", "subject_category"
        ]]
        self.Y = Y

        # Standard image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index: int):
        img_fn = self.X.loc[:, "source_img"][index]
        img_arr = np.array(Image.open(self.image_dir + "/" + img_fn))

        obj_bbox = self.X.loc[:, "object_bbox"][index]
        sub_bbox = self.X.loc[:, "subject_bbox"][index]
        obj_category = self.X.loc[:, "object_category"][index]
        sub_category = self.X.loc[:, "subject_category"][index]

        # Compute crops
        obj_crop = crop_img_arr(img_arr, obj_bbox)
        sub_crop = crop_img_arr(img_arr, sub_bbox)
        union_crop = crop_img_arr(img_arr, union(obj_bbox, sub_bbox))

        # Transform each crop
        image = {
            "obj_crop": self.transform(Image.fromarray(obj_crop)),
            "sub_crop": self.transform(Image.fromarray(sub_crop)),
            "union_crop": self.transform(Image.fromarray(union_crop)),
            "obj_category": obj_category,
            "sub_category": sub_category,
        }
        target = self.Y[index]
        return image, target

    def __len__(self):
        return len(self.X.loc[:, "source_img"])

    def update(self, features, Y):
        self.X = features.copy()
        self.X.index = range(len(self.X))
        self.Y = Y


class VisualRelationClassifier(PerformanceMixin, DiscriminativeModel):
    """Visual Relation Classifier.

    Methods for training and predicting come from DiscriminativeModel base class."""

    def __init__(
        self, pretrained_model, data_path_prefix, soft_labels=True, word_embedding_size=100,
        n_epochs=1, early_stopping=True, warm_start=False,
            checkpoint="../checkpoints/VR_checkpoint.pt", patience=5, lr=1e-3, n_classes=2):

        super().__init__()
        self.pretrained_model = pretrained_model
        self.text_module = WordEmb(
            emb_path=data_path_prefix +
            "word_embeddings/glove.6B." + str(word_embedding_size) + "d.txt"
        ).to(self.device)
        self.concat_module = FlatConcat().to(self.device)
        self.soft_labels = soft_labels
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.warm_start = warm_start
        self.patience = patience
        self.lr = lr
        self.word_embedding_size = word_embedding_size
        self.n_classes = n_classes
        self.checkpoint = checkpoint

        self.reset()

    def extract_concat_features(self, features):
        """Extract image features and word embeddings using pretrained models and concatenate"""

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        feature_extractor = nn.Sequential(
            *list(self.pretrained_model.children())[:-1]
        ).to(self.device)
        sub_features = feature_extractor(features["sub_crop"].to(self.device))
        obj_features = feature_extractor(features["obj_crop"].to(self.device))
        union_features = feature_extractor(features["union_crop"].to(self.device))
        word_embeddings = self.text_module(
            features["obj_category"], features["sub_category"]
        ).to(self.device)

        concatenated_features = self.concat_module(
            sub_features, obj_features, union_features, word_embeddings)

        return concatenated_features

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def reset(self):
        in_features = self.pretrained_model.fc.in_features
        self.linear = nn.Linear(
            in_features * 3 + 2 * self.word_embedding_size, self.n_classes
        ).to(self.device)


class WordEmb(nn.Module):
    """Extract and concat word embeddings for obj and sub categories."""

    def __init__(self, emb_path):
        super().__init__()

        self.word_embs = pd.read_csv(
            emb_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        )

    def _get_wordvec(self, word):
        return self.word_embs.loc[word].values

    def forward(self, obj_category, sub_category):
        obj_emb = self._get_wordvec(obj_category)
        sub_emb = self._get_wordvec(sub_category)
        embs = np.concatenate([obj_emb, sub_emb], axis=1)
        return torch.FloatTensor(embs)


class FlatConcat(nn.Module):
    """Module that flattens and concatenates features"""

    def forward(self, *inputs):
        return torch.cat([input.view(input.size(0), -1) for input in inputs], dim=1)


def crop_img_arr(img_arr, bbox):
    """Crop bounding box from image."""

    # Add dimension for black and white images without RGB colorspace
    if len(img_arr.shape) == 2:
        img_arr = img_arr[:, :, None]

    return img_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], :]


def union(bbox1, bbox2):
    """Create the union of two bounding boxes."""

    y0 = min(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x0 = min(bbox1[2], bbox2[2])
    x1 = max(bbox1[3], bbox2[3])
    return [y0, y1, x0, x1]
