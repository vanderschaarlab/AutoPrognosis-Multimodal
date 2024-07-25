import copy
from types import SimpleNamespace
import numpy as np

from tqdm import tqdm
from src.late_fusion.late_fusion import late_fusion_predict_prob, late_fusion_training
from src.early_fusion.early_fusion_predict_prob import early_fusion_predict_prob
from src.early_fusion.early_fusion_training import early_fusion_training
from src.joint_fusion.joint_fusion_predict_prob import joint_fusion_predict_prob
from src.utils.utils import (
    dict_to_namespace,
    namespace_to_dict,
)
from src.imaging.imaging_training import imaging_training
from src.imaging.imaging_predict_prob import imaging_predict_prob
from src.joint_fusion.joint_fusion_training import joint_fusion_training
from src.tabular.autoprognosis import tabular_predict_prob, tabular_training
import pandas as pd

from src.utils.ensemble import search_weights
from src.utils.metrics import get_metric


class AutoprognosisM:
    def __init__(self, pipeline_config: SimpleNamespace):
        self.pipeline_config = pipeline_config
        self.top_n_idx = None

    def run(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, force=False
    ) -> pd.DataFrame:
        for i, _model_config in tqdm(
            enumerate(self.pipeline_config.configurations), desc="Training"
        ):

            # Combine with global config:
            model_config = dict_to_namespace(
                {
                    **namespace_to_dict(self.pipeline_config.globals),
                    **namespace_to_dict(copy.deepcopy(_model_config)),
                }
            )

            if model_config.type == "imaging":
                # Run imaging training
                imaging_training(model_config, train_df, val_df, force)
            elif model_config.type == "tabular":
                tabular_training(model_config, train_df, force=force)
            elif model_config.type == "joint_fusion":
                joint_fusion_training(model_config, train_df, val_df, force=force)
            elif model_config.type == "early_fusion":
                # Combine with global config:
                model_config.imaging = dict_to_namespace(
                    {
                        **namespace_to_dict(self.pipeline_config.globals),
                        **namespace_to_dict(copy.deepcopy(_model_config.imaging)),
                    }
                )
                early_fusion_training(model_config, train_df, val_df, force)
            elif model_config.type == "late_fusion":
                model_config.imaging = dict_to_namespace(
                    {
                        **namespace_to_dict(self.pipeline_config.globals),
                        **namespace_to_dict(copy.deepcopy(_model_config.imaging)),
                    }
                )

                model_config.tabular = dict_to_namespace(
                    {
                        **namespace_to_dict(self.pipeline_config.globals),
                        **namespace_to_dict(copy.deepcopy(_model_config.tabular)),
                    }
                )
                late_fusion_training(model_config, train_df, val_df, force)
            else:
                raise ValueError(f"Unknown model type: {model_config.type}")

            # update config
            self.pipeline_config.configurations[i] = model_config

    def fit(
        self, df=pd.DataFrame, target_metric: str = "Bal. Acc.", top_n=None
    ) -> np.ndarray:

        prop_val_dfs = []
        gt = None
        task = None

        for model_config in tqdm(
            self.pipeline_config.configurations, desc="Validating"
        ):
            if gt is None:
                gt = pd.get_dummies(
                    df[model_config.target_column],
                    columns=list(sorted(df[model_config.target_column].unique())),
                    dtype=int,
                ).values.argmax(axis=1)

            if task is None:
                task = (
                    "binary"
                    if len(vars(model_config.class_to_idx)) == 2
                    else "multi-class"
                )

            if model_config.type == "imaging":
                # Run imaging prediction
                prop_val_dfs.append(imaging_predict_prob(model_config, df=df))
            elif model_config.type == "tabular":
                # Run tabular prediction
                prop_val_dfs.append(tabular_predict_prob(model_config, df=df))
            elif model_config.type == "joint_fusion":
                # Run joint fusion prediction
                prop_val_dfs.append(joint_fusion_predict_prob(model_config, df=df))
            elif model_config.type == "early_fusion":
                # Run early fusion prediction
                prop_val_dfs.append(early_fusion_predict_prob(model_config, df=df))
            elif model_config.type == "late_fusion":
                # Run late fusion prediction
                prop_val_dfs.append(late_fusion_predict_prob(model_config, df=df))
            else:
                raise ValueError(f"Unknown model type: {model_config.type}")

        if top_n:
            scores = [
                get_metric(
                    preds=prob_val_pred,
                    labels=gt,
                    task=task,
                    metric=target_metric,
                )
                for prob_val_pred in prop_val_dfs
            ]

            # Get the index of the top_n highest scoring models
            self.top_n_idx = np.argsort(scores)[-top_n:]
            prop_val_dfs = [prop_val_dfs[i] for i in self.top_n_idx]

        weights = search_weights(
            val_preds=[p.values for p in prop_val_dfs],
            labels=gt,
            task=task,
            metric_name=target_metric,
        )

        return weights

    def predict(self, df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
        ensemble_prob_dfs = []
        class_names = None  # To store class names
        index = None  # To store the index of the DataFrame

        configurations = self.pipeline_config.configurations
        if self.top_n_idx is not None:
            # Only consider the top_n models
            configurations = [
                self.pipeline_config.configurations[i] for i in self.top_n_idx
            ]

        # Now do the predictions on the test set for those models
        for config in tqdm(configurations, desc="Predicting"):
            if config.type == "imaging":
                # Run imaging prediction
                prob_values_df = imaging_predict_prob(config, df=df)
            elif config.type == "tabular":
                # Run tabular prediction
                prob_values_df = tabular_predict_prob(config, df=df)
            elif config.type == "joint_fusion":
                # Run joint fusion prediction
                prob_values_df = joint_fusion_predict_prob(config, df=df)
            elif config.type == "early_fusion":
                # Run early fusion prediction
                prob_values_df = early_fusion_predict_prob(config, df=df)
            elif config.type == "late_fusion":
                # Run late fusion prediction
                prob_values_df = late_fusion_predict_prob(config, df=df)
            else:
                raise ValueError(f"Unknown model type: {config.type}")

            # Extract class names and index from the columns of the first DataFrame
            if class_names is None:
                class_names = prob_values_df.columns
            if index is None:
                index = prob_values_df.index

            ensemble_prob_dfs.append(prob_values_df)

        # Convert list of DataFrames to numpy array for averaging
        ensemble_prob_array = np.array([df.values for df in ensemble_prob_dfs])

        # Final ensemble predictions: average the probabilities
        ensemble_avg_probs = np.average(ensemble_prob_array, weights=weights, axis=0)

        # Take the argmax and assign the class labels from the columns
        ensemble_predictions = class_names[ensemble_avg_probs.argmax(axis=1)]

        # Create a DataFrame for the final predictions with the same index
        ensemble_predictions_df = pd.DataFrame(
            ensemble_predictions, index=index, columns=["predictions"]
        )
        return ensemble_predictions_df
