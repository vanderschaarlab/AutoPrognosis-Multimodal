import json
from types import SimpleNamespace
import pandas as pd
from pathlib import Path
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file, save_to_file
from src.utils.utils import md5_hash, namespace_to_dict
import shutil


def tabular_training(config: SimpleNamespace, train_df: pd.DataFrame, force=False):
    _train_df = train_df.copy()
    workspace = Path(config.base_experiment_dir) / "autoprognosis"
    workspace.mkdir(parents=True, exist_ok=True)
    study_name_tabular = md5_hash(namespace_to_dict(config))
    model_path = Path(workspace / study_name_tabular / "model.bkp")

    if model_path.exists() and not force:
        print(f"Model already exists at {model_path}. Skipping training.")
        return

    if force:
        shutil.rmtree(workspace / study_name_tabular, ignore_errors=True)

    tabular_train_dataframe = _train_df[config.feature_columns + [config.target_column]]
    # Define study
    tabular_study = ClassifierStudy(
        study_name=study_name_tabular,
        dataset=tabular_train_dataframe,
        target=config.target_column,
        workspace=workspace,
        **namespace_to_dict(config.classifier),
    )
    # run study
    tabular_study.run()

    tabular_model = load_model_from_file(workspace / study_name_tabular / "model.p")
    train_X_tabular = tabular_train_dataframe[
        config.feature_columns + [config.target_column]
    ]
    # Since there is no direct way to get the int->string label mapping, we provide the labels as integers
    classes = sorted(train_X_tabular[config.target_column].unique())
    class_to_idx = {_cls: idx for idx, _cls in enumerate(classes)}
    # apply the mapping
    train_X_tabular[config.target_column] = train_X_tabular[config.target_column].map(
        class_to_idx
    )
    tabular_model.fit(train_X_tabular, tabular_train_dataframe[config.target_column])
    save_to_file(model_path, tabular_model)

    # Save the class as json
    with open(workspace / study_name_tabular / "classes.json", "w") as f:
        json.dump(classes, f)


def tabular_predict_prob(config: SimpleNamespace, df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    study_name_tabular = md5_hash(namespace_to_dict(config))
    workspace = Path(config.base_experiment_dir) / "autoprognosis"
    tabular_model = load_model_from_file(workspace / study_name_tabular / "model.bkp")

    val_dataframe = _df[config.feature_columns + [config.target_column]]

    with open(workspace / study_name_tabular / "classes.json", "r") as f:
        classes = json.load(f)

    class_to_idx = {_cls: idx for idx, _cls in enumerate(classes)}
    val_dataframe[config.target_column] = val_dataframe[config.target_column].map(
        class_to_idx
    )

    val_prob_df = tabular_model.predict_proba(val_dataframe)
    # assign the class names back
    val_prob_df.columns = classes
    # set the index to the original index
    val_prob_df.index = _df[config.index_column]

    return val_prob_df
