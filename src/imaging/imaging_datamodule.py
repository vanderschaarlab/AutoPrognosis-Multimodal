from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.imaging.imaging_dataset import ImageDataset


class LitImageDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        target_column,
        index_column,
        class_to_idx,
        augmentation,
        batch_size,
        num_workers,
        test_df=None,
        transform = None
    ):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ImageDataset(
            df=train_df.set_index(index_column),
            target_column=target_column,
            class_to_idx=class_to_idx,
            transform=self.transform,
            augmentation=augmentation,
        )

        self.val_dataset = ImageDataset(
            df=val_df.set_index(index_column),
            target_column=target_column,
            class_to_idx=class_to_idx,
            transform=self.transform,

        )

        if test_df is not None:
            self.test_dataset = ImageDataset(
                df=test_df.set_index(index_column),
                target_column=target_column,
                class_to_idx=class_to_idx,
                transform=self.transform,
            )
        else:
            self.test_dataset = None

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
