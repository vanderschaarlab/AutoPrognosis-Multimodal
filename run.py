import json

from src.autoprognosis_m import AutoprognosisM
from src.utils.utils import (
    dict_to_namespace,
)
import pandas as pd
import argparse
import json



def main(args):
    pipeline_config = dict_to_namespace(json.load(open(args.config_file, "r")))

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    autoprognosisM = AutoprognosisM(pipeline_config)
    autoprognosisM.run(train_df=train_df, val_df=val_df, force=args.force)
    weights = autoprognosisM.fit(df=val_df, target_metric=args.target_metric)
    ensemble_predictions_df = autoprognosisM.predict(df=test_df, weights=weights)
    return ensemble_predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoprognosis-M")
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="PAD-UFES/config.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/train.csv",
        help="Path to the train csv file",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/val.csv",
        help="Path to the val csv file",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/test.csv",
        help="Path to the test csv file",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force retraining",
    )
    parser.add_argument(
        "-t",
        "--target-metric",
        type=str,
        default="Bal. Acc.",
        choices=["Accuracy", "Bal. Acc.", "AUROC", "F1 Score", "Matt. Corr."],
        help="Target metric for model selection",
    )
    args = parser.parse_args()
    predictions_df = main(args)
    print(predictions_df)
