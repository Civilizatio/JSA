# src/modules/cond_encoders.py
"""Includes the definition and implementation of conditional encoders,
which are used to convert conditional input data (such as images, text, etc.) into a format suitable for processing by Transformer models.

Like the unconditional encoders, `SOSProvider`，only provides the start of sequence token,
while `LabelEncoder` encodes class labels into a format that can be used as input to the model.
"""

# May be changed later to support jsa interface.

import torch
import torch.nn
from src.base.base_cond_encoder import AbstractCondEncoder


class Labelator(AbstractCondEncoder):
    """Net2Net Interface for Class-Conditional Model"""

    def __init__(self, n_classes, quantize_interface=True):
        super().__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c):
        c = c[:, None]
        if self.quantize_interface:
            return c, None, [None, None, c.long()]
        return c


class SOSProvider(AbstractCondEncoder):
    # for unconditional training
    def __init__(self, sos_token):
        super().__init__()
        self.sos_token = sos_token
        
    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1) * self.sos_token # shape (batch_size, 1)
        c = c.long().to(x.device)
        
        return c
