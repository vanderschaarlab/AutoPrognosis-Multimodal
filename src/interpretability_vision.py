import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt

import torch
import numpy as np
from tqdm import tqdm
from src.early_fusion.early_fusion_dataset import TabularDataset
from src.imaging.imaging_dataset import ImageDataset
from src.joint_fusion.joint_fusion_dataset import JointDataset
from src.models import ImageTabularClassifier
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from plotly.subplots import make_subplots
import pandas as pd
from torch.nn import functional as F
from src.utils.utils import assemble_experiment_path, dict_to_namespace, get_hidden_size


class IntegratedGradientsWrapperJoint:
    def __init__(self, model):
        self.model = model
        self.integrated_gradients = IntegratedGradients(self.custom_forward)

    def custom_forward(self, image, tabular_data):
        """
        A wrapper for the forward method that is compatible with Captum.
        """
        return self.model.forward(image, tabular_data).logits

    def compute_integrated_gradients(self, image, tabular_data, target_label):
        # Calculate Integrated Gradients
        attributions = self.integrated_gradients.attribute(
            (image, tabular_data),
            baselines=(torch.zeros_like(image), torch.zeros_like(tabular_data)),
            target=target_label,  # , return_convergence_delta=True
        )

        prediction = F.softmax(self.model.forward(image, tabular_data).logits, dim=-1)

        return attributions, prediction


class IntegratedGradientsWrapperVision:
    def __init__(self, model):
        self.model = model
        self.integrated_gradients = IntegratedGradients(self.custom_forward)

    def custom_forward(self, image):
        """
        A wrapper for the forward method that is compatible with Captum.
        """
        return self.model.forward(image).logits

    def compute_integrated_gradients(self, image, target_label):
        # Calculate Integrated Gradients
        attributions = self.integrated_gradients.attribute(
            image,
            baselines=torch.zeros_like(image),
            target=target_label,
        )

        prediction = F.softmax(self.model.forward(image).logits, dim=-1)
        return attributions, prediction


def load_early_model(config, df: pd.DataFrame):
    # We load the imaging_model and the classifier model into a joint model
    experiment_dir, checkpoint_dir = assemble_experiment_path(config)
    imaging_model, imaging_dataset = load_imaging_model(config.imaging, df)
    hidden_size = get_hidden_size(imaging_model)

    dataset = TabularDataset(
        df=df,
        feature_columns=len(config.feature_columns) + hidden_size,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected 1 checkpoint, found {len(checkpoints)}")
    best_checkpoint = checkpoints[0]

    model_module = importlib.import_module("src.models")
    classifier = getattr(model_module, config.model).load_from_checkpoint(
        best_checkpoint,
        num_labels=dataset.num_classes(),
        num_features=len(config.feature_columns) + hidden_size,
        lr=config.lr,
    )

    # now we have to combine the two models into a joint_fusion model
    joint_fusion_model = ImageTabularClassifier(
        imaging_model=imaging_model,
        tabular_input_size=len(config.feature_columns),
        num_labels=dataset.num_classes(),
        lr=config.lr,
    )

    # And we have to overwrite the classifier with the classifier
    joint_fusion_model.classifier = classifier.model
    # Overwrite the classifier weights
    joint_fusion_model.classifier.load_state_dict(classifier.model.state_dict())

    joint_dataset = JointDataset(
        df=df.set_index(config.index_column),
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    joint_dataset.transform = imaging_model.preprocess

    return joint_fusion_model, joint_dataset


def load_joint_model(config, df: pd.DataFrame):
    experiment_dir, checkpoint_dir = assemble_experiment_path(config)

    dataset = JointDataset(
        df=df.set_index(config.index_column),
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    model_module = importlib.import_module("src.models")
    imaging_model = getattr(model_module, config.model)(
        # both arguments should be ignored
        num_labels=dataset.num_classes(),
        lr=config.lr,
        return_features=True,
    )

    checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected 1 checkpoint, found {len(checkpoints)}")
    best_checkpoint = checkpoints[0]

    model = ImageTabularClassifier.load_from_checkpoint(
        best_checkpoint,
        imaging_model=imaging_model,
        tabular_input_size=len(config.feature_columns),
        num_labels=dataset.num_classes(),
        lr=config.lr,
    )

    dataset.transform = imaging_model.preprocess

    return model, dataset


def load_imaging_model(config, df: pd.DataFrame):
    experiment_dir, checkpoint_dir = assemble_experiment_path(config)

    _df = df.copy().set_index(config.index_column)
    dataset = ImageDataset(
        df=_df,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected 1 checkpoint, found {len(checkpoints)}")
    best_checkpoint = checkpoints[0]

    model_module = importlib.import_module("src.models")
    model = getattr(model_module, config.model).load_from_checkpoint(
        best_checkpoint,
        num_labels=dataset.num_classes(),
        lr=config.lr,
    )

    dataset.transform = model.preprocess

    return model, dataset


def visualize_image_attributes(fig, index, attributions_v, tensor_image_v):
    # Create a custom colormap that starts with black
    custom_cmap = plt.get_cmap("plasma")

    # Convert image to numpy array and normalize for visualization
    image = tensor_image_v.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    # Normalize the image attributions for visualization
    norm = Normalize()
    image_attr_v = attributions_v.squeeze().cpu().detach().numpy()
    image_attr_v = norm(image_attr_v)

    # Create heatmap
    heatmap_v = np.mean(image_attr_v, axis=0)  # Averaging across channels if needed
    # subtract the mean
    heatmap_v -= heatmap_v.mean()
    # cut off negative values
    heatmap_v = np.maximum(heatmap_v, 0)
    # normalize to [0, 1]
    heatmap_v /= heatmap_v.max()
    heatmap_v_rgb = custom_cmap(heatmap_v)[
        :, :, :3
    ]  # Apply colormap and remove alpha channel
    heatmap_v_rgb = (heatmap_v_rgb * 255).astype(np.uint8)

    # Original Image
    fig.add_trace(go.Image(z=image), row=index, col=1)

    # Heatmap Vision
    fig.add_trace(go.Image(z=heatmap_v_rgb), row=index, col=2)


def compute_integrated_gradients_joint(model, dataset, idx, device="cuda"):
    ig_wrapper = IntegratedGradientsWrapperJoint(model.to(device))
    tensor_image, tensor_features, target = dataset.__getitem__(idx)

    # unsqueeze to add batch dimension
    tensor_image = tensor_image.unsqueeze(0).to(device)
    tensor_features = tensor_features.unsqueeze(0).to(device)

    # Compute integrated gradients
    attributions, pred = ig_wrapper.compute_integrated_gradients(
        tensor_image, tensor_features, target
    )

    return (
        attributions,
        tensor_image,
        pred,
    )


def compute_integrated_gradients_vision(model, dataset, idx, device="cuda"):
    ig_wrapper = IntegratedGradientsWrapperVision(model.to(device))
    tensor_image, target = dataset.__getitem__(idx)

    # unsqueeze to add batch dimension
    tensor_image = tensor_image.unsqueeze(0).to(device)

    # Compute integrated gradients
    attributions, pred = ig_wrapper.compute_integrated_gradients(tensor_image, target)

    return attributions, tensor_image, pred


def interpretability_vision(config: SimpleNamespace, df: pd.DataFrame) -> go.Figure:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.type == "imaging":
        model, dataset = load_imaging_model(config, df)
    elif config.type == "joint_fusion":
        model, dataset = load_joint_model(config, df)
    elif config.type == "early_fusion":
        model, dataset = load_early_model(config, df)

    model.eval()

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    fig = make_subplots(
        rows=len(df),
        cols=2,
        subplot_titles=[" "] * (len(df) * 2),  # Initialize with empty titles
    )

    for index, series in tqdm(
        df.iterrows(), total=len(df), desc="Computing attributions"
    ):
        target_class = series["diagnostic"]

        if config.type == "imaging":
            attributions, image, pred = compute_integrated_gradients_vision(
                model, dataset, index, device
            )
        elif config.type in ["joint_fusion", "early_fusion"]:
            attributions_j, image, pred = compute_integrated_gradients_joint(
                model, dataset, index, device
            )
            attributions, _ = attributions_j

        visualize_image_attributes(
            fig,
            index + 1,
            attributions,
            image,
        )
        fig.layout.annotations[index * 2].update(
            text=f"{Path(series[config.index_column]).stem} (GT: {target_class})",
            font=dict(size=10),
        )
        fig.layout.annotations[index * 2 + 1].update(
            text=f"Heatmap (Prediction: {idx_to_class[int(pred.argmax())]})",
            font=dict(size=10),
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        autosize=True,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=20, b=0),  # Adjust margins for tight layout
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoprognosis-M")
    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        help="Path to the config file of a specific model in experiement_dir",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        help="Path to the csv file for visualizing the integrated gradients",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of samples to visualize",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    if args.n:
        df = df[: args.n]
    interpretability_vision(
        dict_to_namespace(json.load(open(args.config_file, "r"))), df
    ).show()
