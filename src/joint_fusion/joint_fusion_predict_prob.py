import importlib
from types import SimpleNamespace
import torch
import pandas as pd
from pytorch_lightning import Trainer
import torch

from src.utils.utils import assemble_experiment_path
from src.joint_fusion.joint_fusion_dataset import JointDataset
from src.models import ImageTabularClassifier
from torch.utils.data import DataLoader


def joint_fusion_predict_prob(args: SimpleNamespace, df: pd.DataFrame) -> pd.DataFrame:
    experiment_dir, checkpoint_dir = assemble_experiment_path(args)

    _df = df.copy().set_index(args.index_column)
    dataset = JointDataset(
        df=_df,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
        class_to_idx=vars(args.class_to_idx),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_module = importlib.import_module("src.models")
    imaging_model = getattr(model_module, args.model)(
        # both arguments should be ignored
        num_labels=dataset.num_classes(),
        lr=args.lr,
        return_features=True,
    )
    model = ImageTabularClassifier(
        imaging_model=imaging_model,
        tabular_input_size=len(args.feature_columns),
        num_labels=dataset.num_classes(),
        lr=args.lr,
    )

    # FIXME: Not ideal but works for now
    dataset.transform = imaging_model.preprocess

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
                model=model,
                ckpt_path=best_checkpoint,
                dataloaders=dataloader,
            )
        ),
        dim=-1,
    ).numpy()

    idx_to_class = dict((v, k) for k, v in dataset.class_to_idx.items())

    prob_columns = [idx_to_class[i] for i in range(prob_pred.shape[1])]

    return pd.DataFrame(
        prob_pred,
        columns=prob_columns,
        index=_df.index,
    )
