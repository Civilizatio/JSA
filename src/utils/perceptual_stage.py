# src/utils/perceptual_stage.py
"""Utilities for stage-aware Perceptual JSA training.

Defines the :class:`PerceptualTrainingStage` enum, which represents the three stages of training used by :class:`PerceptualJSA`. 
This includes methods for parsing stage values from various formats (integers, strings) and provides a clear API for working with training stages in the codebase.
"""

from __future__ import annotations # Allows using the class itself in type annotations within the class definition.

from enum import IntEnum
from typing import Optional, Union


class PerceptualTrainingStage(IntEnum):
    """Three-stage training schedule used by :class:`PerceptualJSA`."""

    DECODER_PRETRAIN = 1
    PRIOR_PRETRAIN = 2
    FULL_EBM = 3

    @classmethod
    def from_value(
        cls, value: Optional[Union["PerceptualTrainingStage", int, str]]
    ) -> Optional["PerceptualTrainingStage"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if not isinstance(value, str):
            raise TypeError(f"Unsupported stage type: {type(value)!r}")

        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "1": cls.DECODER_PRETRAIN,
            "stage1": cls.DECODER_PRETRAIN,
            "stage_1": cls.DECODER_PRETRAIN,
            "decoder": cls.DECODER_PRETRAIN,
            "decoder_pretrain": cls.DECODER_PRETRAIN,
            "reconstruction": cls.DECODER_PRETRAIN,
            "2": cls.PRIOR_PRETRAIN,
            "stage2": cls.PRIOR_PRETRAIN,
            "stage_2": cls.PRIOR_PRETRAIN,
            "prior": cls.PRIOR_PRETRAIN,
            "prior_pretrain": cls.PRIOR_PRETRAIN,
            "3": cls.FULL_EBM,
            "stage3": cls.FULL_EBM,
            "stage_3": cls.FULL_EBM,
            "full": cls.FULL_EBM,
            "full_ebm": cls.FULL_EBM,
            "ebm": cls.FULL_EBM,
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported stage value: {value!r}. Supported values are 1/2/3 or stage aliases."
            )
        return aliases[key]


__all__ = ["PerceptualTrainingStage"] # Explicitly define the public API of this module.
