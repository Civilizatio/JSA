import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn



class PixelLoss(nn.Module):
    """Per-sample pixel reconstruction loss.

    Returns a vector of shape [B] so it can be directly used as an energy term.
    """

    def __init__(
        self,
        norm: str = "l1",
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        norm = norm.lower()
        if norm not in {"l1", "l2", "mse"}:
            raise ValueError(f"Unsupported pixel loss norm: {norm}")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.norm = norm
        self.weight = float(weight)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if self.norm == "l1":
            loss = torch.abs(x - x_hat)
        else:
            loss = (x - x_hat) ** 2

        loss = loss.flatten(1) # [B, C*H*W]
        if self.reduction == "mean":
            loss = loss.mean(dim=1)
        else:
            loss = loss.sum(dim=1)
        return self.weight * loss


class LPIPSPerceptualLoss(nn.Module):
    """LPIPS perceptual loss that returns per-sample energies.

    The LPIPS network itself is frozen, but gradients are allowed to flow through
    the reconstruction input ``x_hat``.
    """

    def __init__(
        self,
        variant: str = "cifar10",
        weight: float = 1.0,
        repeat_gray_to_rgb: bool = True,
    ):
        super().__init__()
        variant = variant.lower()
        from src.modules.losses.lpips import LPIPS, LPIPS_CIFAR10

        if variant in {"cifar10", "lpips_cifar10"}:
            self.lpips = LPIPS_CIFAR10().eval()
        elif variant in {"vgg", "imagenet", "lpips"}:
            self.lpips = LPIPS().eval()
        else:
            raise ValueError(f"Unsupported LPIPS variant: {variant}")

        for param in self.lpips.parameters():
            param.requires_grad = False

        self.variant = variant
        self.weight = float(weight)
        self.repeat_gray_to_rgb = repeat_gray_to_rgb

    def _maybe_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            return x
        if x.shape[1] == 1 and self.repeat_gray_to_rgb:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(
            f"LPIPS expects 1 or 3 input channels, but got shape {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        x = self._maybe_to_rgb(x)
        x_hat = self._maybe_to_rgb(x_hat)
        loss = self.lpips(x.contiguous(), x_hat.contiguous()) # [B, 1, H', W']
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1) # [B]
        return self.weight * loss


class ReconstructionDistortionModel(nn.Module):
    """Distortion model d_theta(x, h).

    This module owns the decoder g_theta(h) and computes a weighted sum of
    pixel-space losses and perceptual losses.
    """

    def __init__(
        self,
        decoder: nn.Module,
        pixel_l1_weight: float = 1.0,
        pixel_l2_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        perceptual_variant: str = "cifar10",
        pixel_reduction: str = "mean",
    ):
        super().__init__()
        self.decoder = decoder

        self.pixel_l1 = (
            PixelLoss(norm="l1", weight=pixel_l1_weight, reduction=pixel_reduction)
            if pixel_l1_weight > 0
            else None
        )
        self.pixel_l2 = (
            PixelLoss(norm="l2", weight=pixel_l2_weight, reduction=pixel_reduction)
            if pixel_l2_weight > 0
            else None
        )
        self.perceptual = (
            LPIPSPerceptualLoss(variant=perceptual_variant, weight=perceptual_weight)
            if perceptual_weight > 0
            else None
        )

    @property
    def latent_dim(self):
        return self.decoder.latent_dim

    @property
    def num_categories(self):
        return self.decoder.num_categories

    @property
    def num_latent_vars(self):
        return getattr(self.decoder, "num_latent_vars", 1)

    def get_last_layer_weight(self):
        if hasattr(self.decoder, "get_last_layer_weight"):
            return self.decoder.get_last_layer_weight()
        return None

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.decode(h)

    def distortion(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        x_hat: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        if x_hat is None:
            if h is None:
                raise ValueError("Either h or x_hat must be provided.")
            x_hat = self.decode(h)

        batch = x.shape[0]
        total = torch.zeros(batch, device=x.device, dtype=x.dtype)
        details: Dict[str, torch.Tensor] = {}

        if self.pixel_l1 is not None:
            pixel_l1 = self.pixel_l1(x, x_hat)
            total = total + pixel_l1
            details["pixel_l1"] = pixel_l1
        else:
            details["pixel_l1"] = torch.zeros(batch, device=x.device, dtype=x.dtype)

        if self.pixel_l2 is not None:
            pixel_l2 = self.pixel_l2(x, x_hat)
            total = total + pixel_l2
            details["pixel_l2"] = pixel_l2
        else:
            details["pixel_l2"] = torch.zeros(batch, device=x.device, dtype=x.dtype)

        if self.perceptual is not None:
            perceptual = self.perceptual(x, x_hat)
            total = total + perceptual
            details["perceptual"] = perceptual
        else:
            details["perceptual"] = torch.zeros(batch, device=x.device, dtype=x.dtype)

        details["distortion"] = total
        if return_details:
            return total, details, x_hat
        return total
