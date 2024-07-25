import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class FinetuningCallback(Callback):
    def __init__(self, fine_tune_lr, warm_up_epochs):
        print(
            f"Fine-tuning the classifier only with lr={fine_tune_lr} until epoch {warm_up_epochs}"
        )
        self.fine_tune_lr = fine_tune_lr
        self.warm_up_epochs = warm_up_epochs
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        # Freeze all layers except the classifier
        trainer.optimizers = [
            torch.optim.AdamW(
                pl_module.model.classifier.parameters(), lr=self.fine_tune_lr
            )
        ]

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch == self.warm_up_epochs:
            # Switch to full model training
            trainer.optimizers = [
                torch.optim.AdamW(pl_module.model.parameters(), lr=pl_module.lr)
            ]
