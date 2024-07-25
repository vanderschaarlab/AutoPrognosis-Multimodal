import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        df,
        feature_columns,
        target_column,
        class_to_idx
    ):
        self.df = df
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def num_features(self):
        return len(self.feature_columns)

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
        label = self.df.iloc[idx][self.target_column]
        features = torch.as_tensor(
            self.df.iloc[idx][self.feature_columns], dtype=torch.float32
        )
        return features, self.class_to_idx[label]
