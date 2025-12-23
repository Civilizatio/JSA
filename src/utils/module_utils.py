# src/utils/module_utils.py
# Utility functions for building neural network modules

import torch.nn as nn


def get_act_func_by_name(name):
    """
    Get activation function by name.
    Args:
        name (str): Name of the activation function (e.g., "relu", "tanh").
    Returns:
        nn.Module: Corresponding activation function.
    """
    name = name.lower()
    mapping = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "softplus": nn.Softplus,
        "swish": nn.SiLU,
        
    }

    if name not in mapping:
        raise ValueError(f"Unsupported activation function: {name}")

    return mapping[name]()


