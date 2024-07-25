import importlib
from types import SimpleNamespace
import torch
import pandas as pd
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
from src.early_fusion.early_fusion_dataset import TabularDataset
from src.utils.utils import assemble_experiment_path
from src.imaging.imaging_predict_features import (
    imaging_predict_features as imaging_predict_features,
)


def early_fusion_predict_prob(
    config: SimpleNamespace, df: pd.DataFrame
) -> pd.DataFrame:
    _df = df.copy()
    features_df = imaging_predict_features(config.imaging, df)
    _df = _df.set_index(config.index_column).join(features_df)

    feature_columns = list(config.feature_columns) + list(features_df.columns)

    experiment_dir, checkpoint_dir = assemble_experiment_path(config)
    dataset = TabularDataset(
        df=_df,
        feature_columns=feature_columns,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_module = importlib.import_module("src.models")
    classifier = getattr(model_module, config.model)(
        num_labels=dataset.num_classes(),
        num_features=dataset.num_features(),
        lr=config.lr,
    )

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
    prob_pred = torch.nn.functional.softmax(
        torch.cat(
            trainer.predict(
                model=classifier,
                ckpt_path=best_checkpoint,
                dataloaders=dataloader,
            )
        ),
        dim=-1,
    ).numpy()

    idx_to_class = dict((v, k) for k, v in dataset.class_to_idx.items())

    # store probabilities in a csv file with the original index
    prob_columns = [idx_to_class[i] for i in range(prob_pred.shape[1])]

    return pd.DataFrame(
        prob_pred,
        columns=prob_columns,
        index=_df.index,
    )
