import shutil
from types import SimpleNamespace
import torch
import random
import numpy as np
import os
import hashlib
from pathlib import Path
import json
from pytorch_lightning import seed_everything


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        # Recursively convert SimpleNamespace objects to dictionaries
        return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        # Recursively apply the function to each item in a list
        return [namespace_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        # Recursively apply the function to each item in a dictionary
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    else:
        # Return the object if it is not a SimpleNamespace, list, or dict
        return obj

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def enable_reproducible_results(seed: int = 42) -> None:
    """Set fixed seed for all the libraries"""
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def md5_hash(args, ignore=[], n=10):
    def sorted_dict(d):
        """Recursively sort a dictionary."""
        if not isinstance(d, dict):
            return d
        return {k: sorted_dict(v) for k, v in sorted(d.items())}
    
    def flatten_dict(d):
        """Flatten a nested dictionary into a list of tuples, sorted by keys."""
        items = []
        for k, v in d.items():
            if isinstance(v, dict):
                items.append((k, flatten_dict(v)))
            else:
                items.append((k, v))
        return items
    
    # Make a copy of the args dictionary
    cp = args.copy()
    
    # Remove ignored keys
    for k in ignore:
        cp.pop(k, None)
    
    # Sort and flatten the dictionary
    sorted_cp = sorted_dict(cp)
    flattened_cp = flatten_dict(sorted_cp)
    
    # Truncate the hash to make it shorter
    return hashlib.md5(str(flattened_cp).encode()).hexdigest()[:n]


def assemble_experiment_path(args, ignore_fields=[]):
    experiments_dir = (
        Path(args.base_experiment_dir)
        / args.model
        / md5_hash(namespace_to_dict(args), ignore_fields)
    )

    checkpoints_dir = (
        Path(args.base_checkpoint_dir)
        / args.model
        / md5_hash(namespace_to_dict(args), ignore_fields)
    )

    return experiments_dir, checkpoints_dir


def create_experiment_paths(
    base_experiment_dir, base_checkpoint_dir, args, force=False, ignore_fields=[]
):
    experiments_dir = (
        Path(base_experiment_dir)
        / args.model
        / md5_hash(namespace_to_dict(args), ignore_fields)
    )
    # if experiments_dir already exists, empty it
    if experiments_dir.exists() and force:
        shutil.rmtree(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(base_checkpoint_dir)
        / args.model
        / md5_hash(namespace_to_dict(args), ignore_fields)
    )
    if checkpoint_dir.exists() and force:
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not (experiments_dir / "config.json").exists():
        with open(experiments_dir / "config.json", "w") as file:
            json.dump(namespace_to_dict(args), file, indent=4, sort_keys=True)

    return experiments_dir, checkpoint_dir


def get_img_aug(size=(224, 224)):
    import imgaug.augmenters as iaa

    return iaa.Sequential(
        [
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Scale(size),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode="symmetric")),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            # noise
            iaa.Sometimes(
                0.1,
                iaa.OneOf(
                    [
                        iaa.Dropout(p=(0, 0.05)),
                        iaa.CoarseDropout(0.02, size_percent=0.25),
                    ]
                ),
            ),
            iaa.Sometimes(
                0.25,
                iaa.OneOf(
                    [
                        iaa.Add((-15, 15), per_channel=0.5),  # brightness
                        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
                    ]
                ),
            ),
        ]
    )


def get_hidden_size(imaging_model):
    if getattr(imaging_model, "hidden_size", None):
        return imaging_model.hidden_size
    elif getattr(imaging_model.model.config, "hidden_size", None):
        return imaging_model.model.config.hidden_size
    elif getattr(imaging_model.model.config, "out_channels", None):
        return imaging_model.model.config.out_channels[-1]
    elif getattr(imaging_model.model.config, "hidden_sizes", None):
        return imaging_model.model.config.hidden_sizes[-1]
    else:
        raise ValueError("Unknown hidden size")
