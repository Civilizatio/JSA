# src/base/base_permuter.py
""" This module defines the `AbstractPermuter` class, which serves as a base class for different types of permuters that can be used in the LatentTransformer model. 
A permuter is responsible for rearranging the order of tokens in a sequence, which can be useful for various purposes such as data augmentation, improving model robustness, or implementing specific permutation strategies.
The `AbstractPermuter` class is an abstract base class that defines the interface for permuters.
Concrete implementations of permuters should inherit from `AbstractPermuter` and implement the `forward` method, which takes an input tensor and a boolean flag indicating whether to reverse the permutation.
"""

import torch.nn as nn

class AbstractPermuter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, reverse=False):
        raise NotImplementedError