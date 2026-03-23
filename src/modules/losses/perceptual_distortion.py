"""Perceptual and pixel-space distortion modules for energy-based JSA training.

The core design principle is simple:

* a decoder maps discrete latent codes ``h`` to a reconstruction ``x_hat``;
* pixel-space and perceptual terms map ``(x, x_hat)`` to a per-sample distortion;
* the distortion is then treated as one component of the joint energy.

The implementation intentionally keeps the interfaces small so they can be reused by
``JointModelCategoricalEnergy``, ``LangevinSampler`` and ``NCGSampler``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.torchvision_compat import ensure_torchvision_compat
from src.base.base_feature_extractor import BaseFeatureExtractor

ensure_torchvision_compat()
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

# Tensor = torch.Tensor


def _infer_batch_repeat_factor(x: Tensor, target_batch: int) -> int:
    if x.shape[0] == target_batch:
        return 1
    if target_batch % x.shape[0] != 0:
        raise ValueError(
            f"Cannot align x batch size {x.shape[0]} to target batch size {target_batch}."
        )
    return target_batch // x.shape[0]


def _match_batch_size(x: Tensor, target_batch: int) -> Tensor:
    """Repeat the batch dimension of x to match target_batch.

    For example, if x has shape [B, ...] and target_batch is N*B, this will return a tensor of shape [N*B, ...] where the original batch is repeated N times.
    If x already has batch size target_batch, it is returned unchanged.
    """
    repeat_factor = _infer_batch_repeat_factor(x, target_batch)
    if repeat_factor == 1:
        return x
    return x.repeat_interleave(repeat_factor, dim=0)


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _as_feature_list(
    outputs: Union[Tensor, Sequence[Tensor], Dict[str, Tensor], OrderedDict],
) -> List[Tensor]:
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    if isinstance(outputs, (dict, OrderedDict)):
        return list(outputs.values())
    raise TypeError(f"Unsupported feature container type: {type(outputs)!r}")


class BaseDistanceLoss(nn.Module):
    """Base class for handling l1 and l2 distance computation and reduction."""

    def __init__(
        self, loss_type: str = "l2", weight: float = 1.0, reduction: str = "mean"
    ):
        super().__init__()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss type: {loss_type!r}")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        self.loss_type = loss_type
        self.reduction = reduction
        self.weight = float(
            weight
        )  # Define weight here, so it can be used by subclasses.

    def _compute_distance(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        if self.loss_type == "l1":
            diff = (lhs - rhs).abs()
        else:
            diff = (lhs - rhs).pow(2)

        dims = tuple(range(1, diff.dim()))  # Reduce over all dimensions except batch.
        if self.reduction == "mean":
            return diff.mean(dim=dims)
        return diff.sum(dim=dims)


class PixelLossTerm(BaseDistanceLoss):
    """Per-sample pixel-space loss term.

    Parameters
    ----------
    loss_type:
        ``"l1"`` or ``"l2"``.
    weight:
        Multiplicative weight applied to the resulting per-sample energy.
    reduction:
        ``"mean"`` normalizes by the number of image elements, while ``"sum"`` keeps the
        raw element-wise accumulation.
    """

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:
        x = _match_batch_size(x, x_hat.shape[0])
        loss = self._compute_distance(x, x_hat)
        return self.weight * loss


class IdentityFeatureExtractor(BaseFeatureExtractor):
    """A tiny feature extractor useful for tests or ablations.

    It simply forwards the input as a single feature map.
    """

    def forward(self, x: Tensor) -> List[Tensor]:
        return [x]


class TorchvisionFeatureExtractor(BaseFeatureExtractor):
    """Frozen torchvision backbone feature extractor.

    This wrapper avoids hard-coding LPIPS and makes it possible to combine multiple feature
    extractors later on. ``weights=None`` keeps the code fully offline-friendly.

    Arguments
    ---------
    backbone:
        The name of the torchvision model to use as the backbone. This should be a valid model name in torchvision.models, such as "vgg16", "resnet18", or "resnet34". The backbone will be frozen and used for feature extraction only.
    return_nodes:
        The names of the layers from which to extract features. If None, uses a default set of layers for the specified backbone. The layers will be returned in the order specified here, and they must be valid nodes in the torchvision model's feature extraction graph.
    weights:
        The name of the weights enum member to use for the backbone, e.g. "IMAGENET1K_V1". If None, uses the default weights for the backbone if available, or randomly initialized weights if not. Setting this to "none" (case-insensitive) will disable pretrained weights and use a randomly initialized backbone.
    resize_to:
        If specified, input images will be resized to this size before being passed to the backbone. This can be an int (for square resizing) or a tuple of (height, width). If None, no resizing is performed.
    input_range:
        The expected input range for the backbone. Can be "neg_one_to_one", "zero_to_one" or "none" (no scaling). This controls how the input images are scaled before being passed to the backbone.
    normalize:
        The type of normalization to apply to the input images. Can be "imagenet" to apply standard ImageNet normalization, or "none" (or None) to skip normalization.
    replicate_gray_to_rgb:
        If True, grayscale inputs with shape [B, 1, H, W] will be replicated to RGB by repeating the single channel three times. This is useful for backbones that expect 3-channel input. If False, grayscale inputs will be passed as-is, which may cause errors if the backbone does not support single-channel input.

    Notes
    -----
    * ``weights`` may trigger a torchvision weight download if the weights are not already cached.
    * grayscale inputs are replicated to RGB by default.
    * the module returns a list of tensors in a deterministic order.
    """

    _DEFAULT_RETURN_NODES = {
        "vgg16": ["features.3", "features.8", "features.15"],
        "resnet18": ["layer1", "layer2", "layer3"],
        "resnet34": ["layer1", "layer2", "layer3"],
    }
    # For `vgg16`, 5 silces are available, but we only use the first 3 by default to save memory and because they are sufficient for good performance in our experiments.
    # The later layers can be enabled by explicitly setting return_nodes to include them. They are ["features.22", "features.29"] for the last two convolutional layers.

    def __init__(
        self,
        backbone: str = "vgg16",
        return_nodes: Optional[Sequence[str]] = None,
        weights: Optional[str] = None,
        resize_to: Optional[Union[int, Sequence[int]]] = None,
        input_range: str = "neg_one_to_one",
        normalize: str = "imagenet",
        replicate_gray_to_rgb: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.resize_to = resize_to
        self.input_range = input_range
        self.normalize = normalize
        self.replicate_gray_to_rgb = replicate_gray_to_rgb

        model_ctor = getattr(
            models, backbone, None
        )  # Look up the model constructor by name. This will be None if the backbone is not found.
        if model_ctor is None:
            raise ValueError(f"Unsupported torchvision backbone: {backbone!r}")

        model_weights = weights
        if isinstance(weights, str):
            if weights.lower() == "none":
                model_weights = None
            else:
                get_model_weights: Callable | None = getattr(
                    models, "get_model_weights", None
                )
                if get_model_weights is not None:
                    try:
                        weights_enum = get_model_weights(model_ctor)
                        model_weights = getattr(weights_enum, weights)
                    except Exception:
                        # Fall back to the raw value. torchvision will raise a clearer error
                        # if the string is not a valid enum member for the requested model.
                        model_weights = weights
        model = model_ctor(weights=model_weights)
        _freeze_module(model)
        model.eval()

        default_nodes = self._DEFAULT_RETURN_NODES.get(backbone)
        if return_nodes is None:
            if default_nodes is None:
                raise ValueError(
                    f"Please explicitly provide return_nodes for backbone {backbone!r}."
                )
            return_nodes = default_nodes

        node_map = OrderedDict(
            (node, f"feat_{i}") for i, node in enumerate(return_nodes)
        )
        self.extractor = create_feature_extractor(model, return_nodes=node_map)
        _freeze_module(self.extractor)
        self.extractor.eval()

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_input(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected image tensor of shape [B, C, H, W], but received {tuple(x.shape)}."
            )

        # Scale input to [0, 1] if it's in [-1, 1].
        if self.input_range == "neg_one_to_one":
            x = (x + 1.0) / 2.0
        elif self.input_range in {"zero_to_one", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported input range mode: {self.input_range!r}")

        # Replicate grayscale to RGB if needed.
        if x.shape[1] == 1 and self.replicate_gray_to_rgb:
            x = x.repeat(1, 3, 1, 1)

        # Resize if requested. We use bilinear interpolation,
        # which is a reasonable default for feature extractors trained on natural images,
        # but this can be disabled by setting resize_to=None.
        if self.resize_to is not None:
            if isinstance(self.resize_to, int):
                size = (self.resize_to, self.resize_to)
            else:
                size = tuple(self.resize_to)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        if self.normalize == "imagenet":
            x = (x - self.imagenet_mean) / self.imagenet_std
        elif self.normalize in {"none", None}:
            pass
        else:
            raise ValueError(f"Unsupported normalize mode: {self.normalize!r}")
        return x

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self._prepare_input(x)
        outputs = self.extractor(x)
        return list(outputs.values())


class FeaturePerceptualLoss(BaseDistanceLoss):
    """Feature-space perceptual loss built from an arbitrary frozen extractor."""

    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        loss_type: str = "l2",
        weight: float = 1.0,
        layer_weights: Optional[Sequence[float]] = None,
        reduction: str = "mean",
        freeze_extractor: bool = True,
    ):
        super().__init__(loss_type, weight, reduction)

        self.feature_extractor = feature_extractor
        self.layer_weights = list(layer_weights) if layer_weights is not None else None

        if freeze_extractor:
            # Make sure the feature extractor is frozen.
            # This is important to prevent gradients from flowing into the backbone,
            # which could cause instability during training and is not the intended use of this module.
            _freeze_module(self.feature_extractor)
            self.feature_extractor.eval()

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:
        x = _match_batch_size(x, x_hat.shape[0])
        feats_real = _as_feature_list(self.feature_extractor(x))
        feats_fake = _as_feature_list(self.feature_extractor(x_hat))

        if len(feats_real) != len(feats_fake):
            raise RuntimeError(
                "Feature extractor returned inconsistent number of feature maps."
            )

        if self.layer_weights is None:
            weights = [1.0] * len(feats_real)
        else:
            if len(self.layer_weights) != len(feats_real):
                raise ValueError(
                    f"layer_weights has length {len(self.layer_weights)}, but extractor returned {len(feats_real)} features."
                )
            weights = self.layer_weights

        total = torch.zeros(x_hat.shape[0], device=x_hat.device, dtype=x_hat.dtype)
        for weight, feat_real, feat_fake in zip(weights, feats_real, feats_fake):
            dist = self._compute_distance(feat_real, feat_fake)
            total = total + float(weight) * dist
        return self.weight * total


class ReconstructionDistortionModel(nn.Module):
    """Combine a decoder with pixel and perceptual distortion terms.

    Arguments
    ---------
    decoder:
        A decoder module that maps latent codes to reconstructions. This should implement a `forward`
        method that takes a tensor of shape [B, ...] and returns a tensor of shape [B, C, H, W]. Optionally, it can also implement `embed` and `decode_from_embeddings` methods for use with NCGSampler.
    pixel_terms:
        A sequence of modules that compute pixel-space distortion terms. Each module should implement a `forward` method that takes two tensors of shape [B, C, H, W] and returns a tensor of shape [B] representing the per-sample distortion. If None or empty, no pixel terms are used.
    perceptual_terms:
        A sequence of modules that compute feature-space perceptual distortion terms. Each module should implement a `forward` method that takes two tensors of shape [B, C, H, W] and returns a tensor of shape [B] representing the per-sample distortion. If None or empty, no perceptual terms are used.
    clamp_output:
        If specified, the output of the decoder will be clamped to this range (min, max). This can be useful if the decoder does not have a built-in output activation and you want to ensure the reconstructions are in a valid pixel range. If None, no clamping is applied.
    """

    def __init__(
        self,
        decoder: nn.Module,
        pixel_terms: Optional[Sequence[nn.Module]] = None,
        perceptual_terms: Optional[Sequence[nn.Module]] = None,
        clamp_output: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.pixel_terms = nn.ModuleList(list(pixel_terms or []))
        self.perceptual_terms = nn.ModuleList(list(perceptual_terms or []))
        self.clamp_output = tuple(clamp_output) if clamp_output is not None else None

        if len(self.pixel_terms) == 0 and len(self.perceptual_terms) == 0:
            # A reasonable offline-safe default.
            self.pixel_terms.append(
                PixelLossTerm(loss_type="l1", weight=1.0, reduction="mean")
            )

    @property
    def latent_dim(self):
        return getattr(self.decoder, "latent_dim", None)

    @property
    def num_categories(self):
        return getattr(self.decoder, "num_categories", None)

    @property
    def embedding_dims(self):
        return getattr(self.decoder, "embedding_dims", None)

    def get_last_layer_weight(self):
        if hasattr(self.decoder, "get_last_layer_weight"):
            return self.decoder.get_last_layer_weight()
        return None

    def decode(self, h: Tensor) -> Tensor:
        x_hat = self.decoder(h)
        if self.clamp_output is not None:
            x_hat = x_hat.clamp(self.clamp_output[0], self.clamp_output[1])
        return x_hat

    def embed_latent(self, h: Tensor) -> Tensor:
        if not hasattr(self.decoder, "embed"):
            raise AttributeError(
                "The decoder does not expose an `embed` method, which is required by NCGSampler."
            )
        return self.decoder.embed(h)

    def decode_from_embedded_latent(self, embedded_h: Tensor) -> Tensor:
        if not hasattr(self.decoder, "decode_from_embeddings"):
            raise AttributeError(
                "The decoder does not expose `decode_from_embeddings`, which is required by NCGSampler."
            )
        x_hat = self.decoder.decode_from_embeddings(embedded_h)
        if self.clamp_output is not None:
            x_hat = x_hat.clamp(self.clamp_output[0], self.clamp_output[1])
        return x_hat

    def distortion_from_reconstruction(self, x: Tensor, x_hat: Tensor):
        x = _match_batch_size(x, x_hat.shape[0])
        total = torch.zeros(x_hat.shape[0], device=x_hat.device, dtype=x_hat.dtype)
        pixel_total = torch.zeros_like(total)
        perceptual_total = torch.zeros_like(total)
        components: Dict[str, Tensor] = {}

        for idx, term in enumerate(self.pixel_terms):
            value = term(x, x_hat)
            pixel_total = pixel_total + value
            components[f"pixel_{idx}"] = value

        for idx, term in enumerate(self.perceptual_terms):
            value = term(x, x_hat)
            perceptual_total = perceptual_total + value
            components[f"perceptual_{idx}"] = value

        total = pixel_total + perceptual_total
        components["pixel_total"] = pixel_total
        components["perceptual_total"] = perceptual_total
        components["distortion_total"] = total

        return {"distortion": total, "components": components}

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        # return_reconstruction: bool = False,
        # return_components: bool = False,
    ):
        x_hat = self.decode(h)
        outputs = self.distortion_from_reconstruction(x, x_hat)

        return {
            "distortion": outputs["distortion"],
            "reconstruction": x_hat,
            "components": outputs["components"],
        }


__all__ = [
    "FeaturePerceptualLoss",
    "IdentityFeatureExtractor",
    "PixelLossTerm",
    "ReconstructionDistortionModel",
    "TorchvisionFeatureExtractor",
]

if __name__ == "__main__":
    from torchvision.models import vgg16
    from torchvision.models.feature_extraction import get_graph_node_names

    model = vgg16(weights="IMAGENET1K_V1")
    train_nodes, eval_nodes = get_graph_node_names(model)
    print("Train nodes:", train_nodes)
    print("Eval nodes:", eval_nodes)
