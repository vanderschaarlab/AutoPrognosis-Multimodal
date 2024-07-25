from types import SimpleNamespace
from src.imaging.imaging_predict_prob import (
    imaging_predict_prob as imaging_predict_prob,
)
from src.imaging.imaging_training import imaging_training as imaging_training
from src.tabular.autoprognosis import tabular_predict_prob, tabular_training
import pandas as pd


def late_fusion_training(
    config: SimpleNamespace, train_df: pd.DataFrame, val_df: pd.DataFrame, force=False
):

    # Run imaging training
    imaging_training(config.imaging, train_df, val_df, force)

    # Run tabular training
    tabular_training(config.tabular, train_df, force)


def late_fusion_predict_prob(config: SimpleNamespace, df: pd.DataFrame) -> pd.DataFrame:
    # Run imaging prediction
    val_prob_df_img = imaging_predict_prob(config.imaging, df)

    # Run tabular prediction
    val_prob_df_tab = tabular_predict_prob(config.tabular, df)

    val_prob_df = (val_prob_df_img + val_prob_df_tab) / 2

    return val_prob_df
