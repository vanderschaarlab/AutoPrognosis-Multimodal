from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class JointDataset(Dataset):
    def __init__(
        self,
        df,
        target_column: str,
        feature_columns: List[str],
        class_to_idx: dict,
        transform=None,
        augmentation=None,
    ):
        self.df = df
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.class_to_idx)

    def weights(self):
        return torch.tensor(
            [
                len(self.df) / (self.df[self.target_column] == cls).sum()
                for cls in self.class_to_idx.keys()
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].name

        image = np.array(Image.open(img_path).convert("RGB"))[np.newaxis, :]

        if self.augmentation:
            image = self.augmentation(images=image)
        if self.transform:
            image = self.transform(image)

        label = self.df.iloc[idx][self.target_column]
        features = torch.as_tensor(
            self.df.iloc[idx][self.feature_columns], dtype=torch.float32
        )
        return image, features, self.class_to_idx[label]
