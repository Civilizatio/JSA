# src/utils/torchvision_compat.py
"""Compatibility helper for environments where torchvision custom ops are missing.

Some evaluation sandboxes ship a torchvision build whose Python package is present but whose
custom C++/CUDA ops are not fully registered. Importing torchvision then fails early while
registering fake kernels such as ``torchvision::nms``.

Registering the operator schema eagerly is enough for the rest of this project because the code
paths used here rely on datasets, transforms, model definitions and utility helpers rather than on
executing the NMS kernel itself.
"""

from __future__ import annotations


_TORCHVISION_LIB = None


def ensure_torchvision_compat() -> None:
    global _TORCHVISION_LIB
    if _TORCHVISION_LIB is not None:
        return
    
    try:
        import torchvision
        return
    except Exception:
        try:
            import torch
            _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
            _TORCHVISION_LIB.define(
                "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
        except Exception:
            # The schema may already exist, or torchvision might already be imported.
            _TORCHVISION_LIB = None


ensure_torchvision_compat() # Ensure compatibility at import time, so that downstream code can safely import torchvision without worrying about the custom op registration status.
__all__ = ["ensure_torchvision_compat"]
