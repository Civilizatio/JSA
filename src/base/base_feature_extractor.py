# src/base/base_feature_extractor.py
"""Base class for feature extractors used in Perceptual JSA.
Defines the :class:`BaseFeatureExtractor` abstract class, which serves as a common interface for all feature extractors used in the codebase.
This class enforces the implementation of a `forward` method that takes an input tensor and returns a list of feature tensors extracted from various layers.
"""

from __future__ import annotations

import abc
import torch.nn as nn
from typing import List
from torch import Tensor

class BaseFeatureExtractor(nn.Module, abc.ABC):
    """Abstract base class for feature extractors."""

    @abc.abstractmethod
    def forward(self, x: Tensor) -> List[Tensor]:
        """Extract features from the input tensor.
        
        Parameters
        ----------
        x: Tensor
            Input tensor, typically an image batch of shape [B, C, H, W].
            
        Returns
        -------
        List[Tensor]:
            A list of feature tensors extracted from various layers.
        """
        pass