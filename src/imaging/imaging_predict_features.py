import importlib
from types import SimpleNamespace
import torch
import pandas as pd
from pytorch_lightning import Trainer
import torch

from src.utils.utils import assemble_experiment_path
from src.imaging.imaging_dataset import ImageDataset


def imaging_predict_features(config: SimpleNamespace, df: pd.DataFrame) -> pd.DataFrame:
    experiment_dir, checkpoint_dir = assemble_experiment_path(config)

    df = df.set_index(config.index_column)

    dataset = ImageDataset(
        df=df,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_module = importlib.import_module("src.models")
    model = getattr(model_module, config.model)(
        num_labels=dataset.num_classes(),
        lr=config.lr,
        return_features=True,
    )

    dataset.transform = model.preprocess

    trainer = Trainer(
        default_root_dir=experiment_dir,
        log_every_n_steps=1,
        deterministic=True,
    )

    checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected 1 checkpoint, found {len(checkpoints)}")
    best_checkpoint = checkpoints[0]

    # Evaluate the best model on the validation set
    # [N, num_classes]
    feature_pred = torch.cat(
        trainer.predict(
            model=model,
            ckpt_path=best_checkpoint,
            dataloaders=dataloader,
        )
    ).numpy()

    return pd.DataFrame(
        feature_pred,
        columns=[f"imaging_feature_{i}" for i in range(feature_pred.shape[1])],
        index=df.index,
    )
