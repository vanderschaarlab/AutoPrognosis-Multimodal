import importlib
from pathlib import Path
from types import SimpleNamespace
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from src.finetuning_callback import FinetuningCallback
from src.joint_fusion.joint_fusion_datamodule import JointDataModule
from src.models import ImageTabularClassifier
from src.utils.utils import (
    create_experiment_paths,
    enable_reproducible_results,
    get_img_aug,
)
import pandas as pd


def joint_fusion_training(
    config: SimpleNamespace, train_df: pd.DataFrame, val_df: pd.DataFrame, force=False
):
    _train_df, _val_df = train_df.copy(), val_df.copy()
    enable_reproducible_results(config.seed)
    # also stores config
    experiments_dir, checkpoint_dir = create_experiment_paths(
        Path(config.base_experiment_dir),
        Path(config.base_checkpoint_dir),
        config,
        force,
    )

    print(f"Experiment dir: {experiments_dir}")
    if len(list(checkpoint_dir.rglob("*.ckpt"))) and not force:
        print(f"Skipping {experiments_dir} as it already exists")
        return

    datamodule = JointDataModule(
        train_df=_train_df,
        val_df=_val_df,
        target_column=config.target_column,
        index_column=config.index_column,
        feature_columns=config.feature_columns,
        class_to_idx=vars(config.class_to_idx),
        augmentation=get_img_aug() if config.augmentation else None,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model_module = importlib.import_module("src.models")
    imaging_model = getattr(model_module, config.model)(
        # both arguments should be ignored
        num_labels=datamodule.num_classes(),
        weights=(datamodule.weights() if config.weighted_loss else None),
        lr=config.lr,
        return_features=True,
    )
    model = ImageTabularClassifier(
        imaging_model=imaging_model,
        tabular_input_size=datamodule.tabular_input_size(),
        num_labels=datamodule.num_classes(),
        weights=(datamodule.weights() if config.weighted_loss else None),
        lr=config.lr,
    )

    datamodule.train_dataset.transform = imaging_model.preprocess
    datamodule.val_dataset.transform = imaging_model.preprocess

    # Setting up WandbLogger
    callbacks = [
        EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.patience,
            mode=config.monitor_mode,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir / "checkpoints/",
            filename="{epoch}-{val_balanced_acc:.3f}-{val_acc:.3f}-{val_loss:.3f}-{train_acc:.3f}-{train_loss:.3f}",
            monitor=config.monitor_metric,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config.fine_tune_lr and config.warm_up_epochs:
        callbacks.append(
            FinetuningCallback(
                fine_tune_lr=config.fine_tune_lr,
                warm_up_epochs=config.warm_up_epochs,
            )
        )
    trainer = Trainer(
        default_root_dir=experiments_dir,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        min_epochs=config.min_epochs,
        gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(model, datamodule=datamodule)
