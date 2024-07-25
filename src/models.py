from types import SimpleNamespace
from typing import OrderedDict
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
    AutoModelForImageClassification,
    ResNetForImageClassification,
    Dinov2ForImageClassification,
    ViTMAEModel,
)

from pytorch_lightning import LightningModule

from src.utils.utils import get_hidden_size


class BaseClassifier(LightningModule):
    def __init__(
        self, model, num_labels, weights=None, return_features=False, **kwargs
    ):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.return_features = return_features
        # assign all kwargs to self
        {setattr(self, k, v) for k, v in kwargs.items()}
        if weights is None: 
            weights = torch.ones(num_labels)
        self.loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.validation_step_outputs = []

        nan_layers = [
            name for name, param in self.named_parameters() if torch.isnan(param).any()
        ]
        if nan_layers:
            raise ValueError(f"Model has NaNs in layers: {nan_layers}")

    def forward(self, x, return_features=None):
        if return_features is None:
            return_features = self.return_features

        output = self.model(x, output_hidden_states=return_features)

        if return_features:
            hidden_states = output.hidden_states[-1]
            if hidden_states.dim() == 3 and isinstance(
                self.model, (ViTForImageClassification, Dinov2ForImageClassification)
            ):
                hidden_states = hidden_states[:, 0, :]
            elif hidden_states.dim() == 3 and isinstance(self.model, MAEWrapper):
                hidden_states = hidden_states.mean(dim=1)
            elif hidden_states.dim() == 4:
                hidden_states = hidden_states.mean(dim=[2, 3])
            else:
                raise ValueError(
                    f"Unknown hidden states for model {self.model.__class__.__name__} shape: {hidden_states.shape}"
                )
            return SimpleNamespace(logits=output.logits, hidden_states=hidden_states)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs.logits, labels)
        if loss.isnan():
            raise ValueError("NaN loss")

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        acc = accuracy_score(labels.cpu(), outputs.logits.detach().argmax(axis=1).cpu())
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs.logits, labels)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        acc = accuracy_score(labels.cpu(), outputs.logits.argmax(axis=1).cpu())
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.validation_step_outputs.append(
            {
                "preds": outputs.logits.argmax(axis=1),
                "labels": labels.cpu(),
            }
        )
        return outputs

    def on_validation_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        balanced_acc = balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log(
            "val_balanced_acc", balanced_acc, on_epoch=True, prog_bar=True, logger=True
        )
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        return outputs.hidden_states if self.return_features else outputs.logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class Classifier(BaseClassifier):
    def __init__(
        self,
        num_features,
        num_labels,
        activation_fn=torch.nn.ReLU,
        dropout_prob=0.5,
        **kwargs,
    ):
        model = torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            activation_fn(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(32, num_labels),
        )
        super().__init__(model, num_labels, **kwargs)
        self.init_weights()

    def init_weights(self):
        for module in self.model:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, output_hidden_states=False):
        if output_hidden_states:
            raise ValueError("Hidden states not supported for this model")
        logits = self.model(x)
        return SimpleNamespace(**dict(logits=logits))


class ViT(BaseClassifier):
    def __init__(self, model_name, num_labels, **kwargs):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        super().__init__(model, num_labels, **kwargs)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


class ViTBase(ViT):
    def __init__(self, **kwargs):
        super().__init__("google/vit-base-patch16-224", **kwargs)


class ViTLarge(ViT):
    def __init__(self, **kwargs):
        super().__init__("google/vit-large-patch16-224", **kwargs)


class ResNet(BaseClassifier):
    def __init__(self, model_name, num_labels, **kwargs):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = ResNetForImageClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        super().__init__(model, num_labels, **kwargs)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


class ResNet18(ResNet):
    def __init__(self, **kwargs):
        super().__init__("microsoft/resnet-18", **kwargs)


class ResNet34(ResNet):
    def __init__(self, **kwargs):
        super().__init__("microsoft/resnet-34", **kwargs)


class ResNet50(ResNet):
    def __init__(self, **kwargs):
        super().__init__("microsoft/resnet-50", **kwargs)


class ResNet101(ResNet):
    def __init__(self, **kwargs):
        super().__init__("microsoft/resnet-101", **kwargs)


class ResNet152(ResNet):
    def __init__(self, **kwargs):
        super().__init__("microsoft/resnet-152", **kwargs)


class EfficientNet(BaseClassifier):
    def __init__(self, model_name, num_labels, **kwargs):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        super().__init__(model, num_labels, **kwargs)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


class EfficientNetB0(EfficientNet):
    def __init__(self, **kwargs):
        super().__init__("google/efficientnet-b0", **kwargs)


class EfficientNetB1(EfficientNet):
    def __init__(self, **kwargs):
        super().__init__("google/efficientnet-b1", **kwargs)


class EfficientNetB2(EfficientNet):
    def __init__(self, **kwargs):
        self.hidden_size = 352
        super().__init__("google/efficientnet-b2", **kwargs)


class EfficientNetB3(EfficientNet):
    def __init__(self, **kwargs):
        self.hidden_size = 384
        super().__init__("google/efficientnet-b3", **kwargs)


class EfficientNetB4(EfficientNet):
    def __init__(self, **kwargs):
        self.hidden_size = 448
        super().__init__("google/efficientnet-b4", **kwargs)


class EfficientNetB5(EfficientNet):
    def __init__(self, **kwargs):
        self.hidden_size = 512
        super().__init__("google/efficientnet-b5", **kwargs)


class EfficientNetB6(EfficientNet):
    def __init__(self, **kwargs):
        super().__init__("google/efficientnet-b6", **kwargs)


class EfficientNetB7(EfficientNet):
    def __init__(self, **kwargs):
        super().__init__("google/efficientnet-b7", **kwargs)


class MobileNet(BaseClassifier):
    def __init__(self, model_name, num_labels, **kwargs):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        super().__init__(model, num_labels, **kwargs)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


class MobileNetV2(MobileNet):
    def __init__(self, **kwargs):
        super().__init__("google/mobilenet_v2_1.0_224", **kwargs)
        self.model.config.hidden_size = 320


class DinoV2(BaseClassifier):
    def __init__(self, model_name, num_labels, **kwargs):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = Dinov2ForImageClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        super().__init__(model, num_labels, **kwargs)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


class DinoV2Small(DinoV2):
    def __init__(self, **kwargs):
        super().__init__("facebook/dinov2-small", **kwargs)


class DinoV2Base(DinoV2):
    def __init__(self, **kwargs):
        super().__init__("facebook/dinov2-base", **kwargs)


class DinoV2Large(DinoV2):
    def __init__(self, **kwargs):
        super().__init__("facebook/dinov2-large", **kwargs)


class MAEWrapper(torch.nn.Module):
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        self.config = self.model.config

    def forward(self, x, output_hidden_states=False):
        output = self.model(x, output_hidden_states=output_hidden_states)
        logits = self.classifier(output.last_hidden_state.mean(dim=1))
        res = vars(output)
        res["logits"] = logits
        return SimpleNamespace(**res)


class VitMAE(BaseClassifier):
    def __init__(self, model_name, num_labels, **kwargs):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        model = MAEWrapper(ViTMAEModel.from_pretrained(model_name), num_labels)
        super().__init__(model, num_labels, **kwargs)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


class VitMAEBase(VitMAE):
    def __init__(self, **kwargs):
        super().__init__("facebook/vit-mae-base", **kwargs)


class VitMAELarge(VitMAE):
    def __init__(self, **kwargs):
        super().__init__("facebook/vit-mae-large", **kwargs)


class ImageTabularClassifier(LightningModule):
    def __init__(
        self,
        imaging_model,
        tabular_input_size,
        num_labels,
        weights=None,
        activation_fn=torch.nn.ReLU,
        dropout_prob=0.5,
        **kwargs,
    ):
        super().__init__()
        self.imaging_model = imaging_model
        self.num_labels = num_labels

        imaging_model_hidden_size = get_hidden_size(self.imaging_model)
        combined_input_size = imaging_model_hidden_size + tabular_input_size

        hidden_size = 32
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(combined_input_size, hidden_size),
            activation_fn(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size, num_labels),
        )

        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("imaging_model", self.imaging_model),
                    ("classifier", self.classifier),
                ]
            )
        )
        if weights is None:
            weights = torch.ones(num_labels)
        self.loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.validation_step_outputs = []
        {setattr(self, k, v) for k, v in kwargs.items()}

    def forward(self, image, tabular_data):
        imaging_output = self.imaging_model(image, return_features=True)
        hidden_state = imaging_output.hidden_states

        combined_features = torch.cat((hidden_state, tabular_data), dim=1)
        logits = self.classifier(combined_features)

        return SimpleNamespace(logits=logits, hidden_states=hidden_state)

    def training_step(self, batch, batch_idx):
        image, tabular_data, labels = batch
        outputs = self.forward(image, tabular_data)
        loss = self.loss(outputs.logits, labels)
        if loss.isnan():
            raise ValueError("NaN loss")

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        acc = accuracy_score(labels.cpu(), outputs.logits.detach().argmax(axis=1).cpu())
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, tabular_data, labels = batch
        outputs = self.forward(image, tabular_data)
        loss = self.loss(outputs.logits, labels)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        acc = accuracy_score(labels.cpu(), outputs.logits.argmax(axis=1).cpu())
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.validation_step_outputs.append(
            {
                "preds": outputs.logits.argmax(axis=1),
                "labels": labels.cpu(),
            }
        )
        return outputs

    def on_validation_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        balanced_acc = balanced_accuracy_score(labels.cpu(), preds.cpu())
        self.log(
            "val_balanced_acc", balanced_acc, on_epoch=True, prog_bar=True, logger=True
        )
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        image, tabular_data, labels = batch
        outputs = self.forward(image, tabular_data)
        return outputs.logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
