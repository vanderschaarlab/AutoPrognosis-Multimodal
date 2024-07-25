import numpy as np
import pandas as pd
from typing import List
from autoprognosis.explorers.core.optimizer import EnsembleOptimizer

from src.utils.metrics import get_metric

EPS = 1e-8


def search_weights(
    val_preds: List,
    labels: pd.Series,
    task: str,
    metric_name: str,
    name: str = "test",
    seed: int = 42,
):
    """
    A function to find the optimal weights for an ensemble based on a set of predictions (logits) on a validation set.

    Parameters
    ----------
    val_preds : List
        A List containing the predictions (logits) on the validation set.
        The list should have n elements, where each is the predictions (logits) of a different model.
    labels : pd.Series
        Labels for the validation set
    task : str
        Task type.
        Options are:
            - 'binary' for binary classification
            - 'multi-class' for multi-class classification
    metric_name : str
        Metric used to determine the best ensemble.
        Options are:
            - 'Accuracy', 'AUROC', 'F1 Score', 'Matt. Corr.' for the Cancer (binary classification) task
            - 'Accuracy', 'Bal. Acc.', 'AUROC', 'F1 Score' for the Lesion (multi-class classification) task
    name : str, Default = 'test'
        Name of the study, to be used in the caches.
    seed : int, Default = 42
        Random seed
    """

    def evaluate(weights: List) -> float:
        if sum(weights) == 0:
            # print('Weights sum to 0')
            return 0
        avg_preds = np.average(val_preds, weights=weights, axis=0)
        score = get_metric(avg_preds, labels, task, metric_name)
        return score

    study = EnsembleOptimizer(
        study_name=name,
        ensemble_len=len(val_preds),
        evaluation_cbk=evaluate,
        optimizer_type="bayesian",
        n_trials=50,
        timeout=60,
        random_state=seed,
    )

    best_score, selected_weights = study.evaluate()
    weights = []
    for idx in range(len(val_preds)):
        weights.append(selected_weights[f"weight_{idx}"])
    weights = weights / (np.sum(weights) + EPS)

    return weights
