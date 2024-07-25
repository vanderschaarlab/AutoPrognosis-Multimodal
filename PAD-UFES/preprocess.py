import argparse
import numpy as np
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm


def download_progress_hook(count, block_size, total_size):
    """
    A hook to report the progress of a download. This is mostly intended for use
    with `urllib.request.urlretrieve`.

    Args:
        count (int): The current block count.
        block_size (int): The size of each block.
        total_size (int): The total size of the download.
    """
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading: {percent}%", end="")


def download_and_extract_images(url, data_dir=Path("data")):
    # Ensure the destination directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    dest_image_dir = data_dir / "images"

    # Define the path for the downloaded zip file
    zip_path = data_dir / "dataset.zip"

    # Download the zip file
    urllib.request.urlretrieve(url, zip_path, reporthook=download_progress_hook)

    # Unzip the downloaded file into the destination directory
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Find and unzip all zip files in the 'images' subdirectory of the destination directory
    for image_zip_file in dest_image_dir.rglob("*.zip"):
        with zipfile.ZipFile(image_zip_file, "r") as zip_ref:
            zip_ref.extractall(dest_image_dir)

    # Remove the original zip file
    zip_path.unlink()

    # Ensure the destination directory exists
    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Move all png files from data/images to data/tmp
    for filename in dest_image_dir.rglob("*.png"):
        shutil.move(str(filename), str(tmp_dir / filename.name))

    # Remove the original data/images directory if it's empty
    shutil.rmtree(dest_image_dir)
    # If you need to move data/tmp back to data/images
    tmp_dir.rename(dest_image_dir)


def resize_images(data_dir, target_dims):
    # Convert string data_dir to Path object if necessary
    data_dir = Path(data_dir)

    # Load the images
    image_dir = data_dir / "images"

    for image_path in tqdm(list(image_dir.rglob("*.png"))):
        # Load the image
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        # Resize the image
        img_resized = cv2.resize(img, target_dims)

        # Save the resized image, overwriting the original
        cv2.imwrite(str(image_path), img_resized)


def prepare_data(data_dir):
    # Load data
    file_path = data_dir / "metadata.csv"

    data = pd.read_csv(file_path)

    # Replace UNK with nan
    data = data.replace("UNK", np.nan)

    clin_ = ["img_id", "diagnostic", "patient_id", "lesion_id", "biopsed"]
    # All features
    clin_feats = [
        "smoke",
        "drink",
        "background_father",
        "background_mother",
        "age",
        "pesticide",
        "gender",
        "skin_cancer_history",
        "cancer_history",
        "has_piped_water",
        "has_sewage_system",
        "fitspatrick",
        "region",
        "diameter_1",
        "diameter_2",
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
    ]
    # Selected (based on missingness) - These have low missingness & were actually recorded as unknown rather than missing
    clin_feats = [
        "age",
        "region",
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
    ]

    ###
    # CREDIT - https://github.com/paaatcha/MetaBlock/blob/main/benchmarks/pad/preprocess/prepare_data.py
    data_csv = pd.read_csv(file_path).fillna("EMPTY")
    new_cli_cols = list()
    for c in clin_feats:
        if c in ["age", "diameter_1", "diameter_2"]:
            new_cli_cols += [c]
            continue
        vals = [c + "_" + str(v) for v in data_csv[c].unique()]
        try:
            vals.remove(c + "_EMPTY")
        except:
            pass
        new_cli_cols += vals

    new_df = {c: list() for c in new_cli_cols}

    for idx, row in data_csv.iterrows():
        _aux = list()
        _aux_in = list()
        for col in clin_feats:
            data_row = row[col]

            if data_row == "EMPTY":
                pass
            elif col in ["age", "diameter_1", "diameter_2"]:
                _aux_in.append(col)
                new_df[col].append(data_row)
                continue
            else:
                _aux.append(col + "_" + str(data_row))

        for x in new_df:
            if x in _aux:
                new_df[x].append(1)
            elif x not in _aux_in:
                new_df[x].append(0)

    new_df = pd.DataFrame.from_dict(new_df)
    for col in clin_:
        new_df[col] = data_csv[col]

    data = new_df

    # Drop redundant features
    data = data.drop(
        columns=[
            "itch_UNK",
            "grew_UNK",
            "hurt_UNK",
            "changed_UNK",
            "bleed_UNK",
            "elevation_UNK",
        ]
    )
    data = data.drop(columns=["biopsed"])

    # Preprocessing label - Binary
    binary_cancer_classes = {
        "BCC": "malignant",
        "ACK": "benign",
        "NEV": "benign",
        "SEK": "benign",
        "SCC": "malignant",
        "MEL": "malignant",
    }

    data["malignancy"] = data["diagnostic"].apply(lambda x: binary_cancer_classes[x])

    return data


def generate_folds(data_df, split_dir, data_dir):
    """
    Given the dataset folder, create a csv file for each split (train, val, test) containing the image paths and the corresponding labels.

    """
    image_colum = "img_id"

    train_split_idxs_per_fold = [
        eval(line) for line in open(split_dir / f"cv_train_img_idxs.txt", "r")
    ]

    val_split_idxs_per_fold = [
        eval(line) for line in open(split_dir / f"cv_val_img_idxs.txt", "r")
    ]

    test_split_idxs_per_fold = [
        eval(line) for line in open(split_dir / f"cv_test_img_idxs.txt", "r")
    ]

    # Load the metadata
    metadata = data_df.set_index(image_colum)

    # Create the splits for each fold and store them in the dataset folder as fold_{fold_idx}/{split_name}_idxs.csv
    for fold_idx, (train_idxs, val_idxs, test_idxs) in enumerate(
        zip(
            train_split_idxs_per_fold, val_split_idxs_per_fold, test_split_idxs_per_fold
        )
    ):
        (data_dir / "folds" / f"fold_{fold_idx}").mkdir(exist_ok=True, parents=True)
        for split_name, split_idxs in zip(
            ["train", "val", "test"], [train_idxs, val_idxs, test_idxs]
        ):
            split_df = metadata.loc[split_idxs]

            split_df.reset_index(inplace=True)
            split_df[image_colum] = split_df[image_colum].apply(
                lambda x: data_dir / "images" / x
            )

            split_df.set_index(image_colum, inplace=True)

            split_df.to_csv(
                data_dir / "folds" / f"fold_{fold_idx}/{split_name}.csv", index=True
            )


def main(data_dir, split_dir, resize, force):
    if data_dir.exists() and not force:
        print(f"Skipping {data_dir} as it already exists. Use --force to overwrite.")
        return

    if data_dir.exists():
        shutil.rmtree(data_dir)

    print("Downloading and extracting images...")
    download_and_extract_images(
        "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip",
        data_dir=data_dir,
    )
    print()
    print("Images downloaded and extracted.")

    if resize:
        target_dims = (448, 448)
        print(f"Resizing images to {target_dims}...")
        resize_images(data_dir, target_dims)
        print("Images resized.")

    print("Preparing data...")
    prepared_data_df = prepare_data(data_dir)
    print("Data prepared.")

    # Generate fold csv files
    print("Generating folds...")
    generate_folds(prepared_data_df, split_dir=split_dir, data_dir=data_dir)
    print("Folds generated.")

    print("====================================")
    print("Data preparation complete.")
    print("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data directories.")
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data"), help="Directory for the data"
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=Path("PAD-UFES"),
        help="Where we expect the files which define the splits",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize images to 448x448 to speed up processing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the data directory if it already exists",
    )

    args = parser.parse_args()

    main(args.data_dir, args.split_dir, args.resize, args.force)
