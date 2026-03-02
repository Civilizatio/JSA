# src/utils/module_utils.py
import torch
import torch.nn as nn
import importlib
import yaml
from jsonargparse import ArgumentParser


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


def dynamic_import(class_path: str):
    """
    Dynamically import a class from a given module path.

    Args:
        class_path (str): The full path to the class, e.g., "src.models.jsa.JSA".

    Returns:
        type: The imported class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model_from_checkpoint(config_path: str, checkpoint_path: str):
    """Load a model from a checkpoint using a configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        checkpoint_path (str): Path to the model checkpoint file.
    Returns:
        nn.Module: The loaded model.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    model_class_path = model_config.get("class_path", {})
    model_init_args = model_config.get("init_args", {})

    try:
        ModelClass = dynamic_import(model_class_path)
        print(f"Dynamically imported model class: {ModelClass}")
    except Exception as e:
        print(f"Error importing model class from path '{model_class_path}': {e}")
        raise

    parser = ArgumentParser()
    parser.add_class_arguments(
        ModelClass, "model", fail_untyped=False
    )  # Important: setting fail_untyped=False to allow instantiation without type annotations

    cfg = parser.parse_object({"model": model_init_args})
    model = parser.instantiate_classes(cfg).model

    # Load the model parameters into the model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    model.freeze()
    print(f"Model parameters loaded and model set to eval mode.")

    return model
