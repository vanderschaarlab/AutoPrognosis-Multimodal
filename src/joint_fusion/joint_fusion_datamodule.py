from typing import List
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.joint_fusion.joint_fusion_dataset import JointDataset


class JointDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        target_column,
        index_column,
        feature_columns: List[str],
        class_to_idx,
        augmentation,
        batch_size,
        num_workers,
        test_df=None,
        transform=None
    ):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_columns = feature_columns

        self.train_dataset = JointDataset(
            df=train_df.set_index(index_column),
            feature_columns=self.feature_columns,
            target_column=target_column,
            class_to_idx=class_to_idx,
            transform=self.transform,
            augmentation=augmentation,
        )

        self.val_dataset = JointDataset(
            df=val_df.set_index(index_column),
            feature_columns=self.feature_columns,
            target_column=target_column,
            class_to_idx=class_to_idx,
            transform=self.transform,
            augmentation=augmentation,
        )

        if test_df is not None:
            self.test_dataset = JointDataset(
                df=test_df.set_index(index_column),
                feature_columns=self.feature_columns,
                target_column=target_column,
                class_to_idx=class_to_idx,
                transform=self.transform,
                augmentation=augmentation,
            )
        else:
            self.test_dataset = None

    def tabular_input_size(self):
        return len(self.feature_columns)

    def num_classes(self):
        return self.train_dataset.num_classes()

    def weights(self):
        return self.train_dataset.weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
