from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.early_fusion.early_fusion_dataset import TabularDataset


class LitTabularDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        target_column,
        index_column,
        feature_columns,
        class_to_idx,
        batch_size,
        num_workers,
        test_df=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = TabularDataset(
            df=train_df.set_index(index_column),
            feature_columns=feature_columns,
            target_column=target_column,
            class_to_idx=class_to_idx,
        )

        self.val_dataset = TabularDataset(
            df=val_df.set_index(index_column),
            feature_columns=feature_columns,
            target_column=target_column,
            class_to_idx=class_to_idx,
        )

        if test_df is not None:
            self.test_dataset = TabularDataset(
                df=test_df.set_index(index_column),
                feature_columns=feature_columns,
                target_column=target_column,
                class_to_idx=class_to_idx,
            )
        else:
            self.test_dataset = None

    def num_features(self):
        return self.train_dataset.num_features()

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
