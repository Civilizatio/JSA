# src/base/base_cond_encoder.py
""" This module defines the `AbstractEncoder` class, which serves as a base class for different types of conditional encoders that can be used in the LatentTransformer model.
A conditional encoder is responsible for encoding conditional input data (such as images, text, etc.) into a format suitable for processing by Transformer models.
The `AbstractEncoder` class is an abstract base class that defines the interface for conditional encoders.
Concrete implementations of conditional encoders should inherit from `AbstractEncoder` and implement the `encode` method, which takes the input data and returns the encoded representation.
"""

import torch.nn as nn 

class AbstractCondEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError