import importlib
from pathlib import Path
from types import SimpleNamespace
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from src.early_fusion.early_fusion_datamodule import LitTabularDataModule
from src.utils.utils import (
    create_experiment_paths,
    enable_reproducible_results,
)
from src.imaging.imaging_training import imaging_training as imaging_training
from src.imaging.imaging_predict_features import (
    imaging_predict_features as imaging_predict_features,
)
import pandas as pd


def early_fusion_training(
    config: SimpleNamespace, train_df: pd.DataFrame, val_df: pd.DataFrame, force=False
):
    _train_df, _val_df = train_df.copy(), val_df.copy()

    imaging_training(config.imaging, _train_df, _val_df, force)

    enable_reproducible_results(config.seed)

    train_features_df = imaging_predict_features(config.imaging, df=_train_df)
    features_train_df = _train_df.set_index(config.index_column).join(train_features_df)

    val_features_df = imaging_predict_features(config.imaging, df=_val_df)
    features_val_df = _val_df.set_index(config.imaging.index_column).join(
        val_features_df
    )

    feature_columns = list(config.feature_columns) + list(train_features_df.columns)

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

    datamodule = LitTabularDataModule(
        train_df=features_train_df.reset_index(),
        val_df=features_val_df.reset_index(),
        target_column=config.target_column,
        index_column=config.index_column,
        class_to_idx=vars(config.class_to_idx),
        feature_columns=feature_columns,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model_module = importlib.import_module("src.models")
    model = getattr(model_module, config.model)(
        num_labels=datamodule.num_classes(),
        weights=(datamodule.weights() if config.weighted_loss else None),
        num_features=datamodule.num_features(),
        lr=config.lr,
    )

    callbacks = [
        EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.patience,
            mode=config.monitor_mode,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_balanced_acc:.3f}-{val_acc:.3f}-{val_loss:.3f}-{train_acc:.3f}-{train_loss:.3f}",
            monitor=config.monitor_metric,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = Trainer(
        default_root_dir=experiments_dir,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        min_epochs=config.min_epochs,
        gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(model, datamodule=datamodule)
