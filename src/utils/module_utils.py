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
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Error importing '{class_path}': {e}")


def instantiate_from_config(config: dict):
    """
    Instantiate an object from a configuration dictionary.

    Args:
        config (dict): A dictionary containing 'class_path' and 'init_args'.

    Returns:
        object: An instance of the specified class initialized with the given arguments.
    """
    class_path = config.get("class_path")
    init_args = config.get("init_args", {})

    if not class_path:
        raise ValueError("Configuration must include 'class_path'.")

    try:
        ClassType = dynamic_import(class_path)
    except Exception as e:
        raise RuntimeError(f"Error initializing from config: {e}")

    # Using jsonargparse to handle nested configurations and instantiation
    parser = ArgumentParser()
    parser.add_class_arguments(ClassType, "model", fail_untyped=False)

    try:
        cfg = parser.parse_object({"model": init_args})
        instance = parser.instantiate_classes(cfg).model
    except Exception as e:
        raise RuntimeError(
            f"Error instantiating class '{class_path}' with arguments {init_args}: {e}"
        )
        
    return instance

def load_from_config(config: dict, ckpt_path: str=None, freeze: bool = True):
    """
    Initialize an object from a configuration dictionary and load its state from a checkpoint.

    Args:
        config (dict): A dictionary containing 'class_path' and 'init_args'.
        ckpt_path (str): Path to the model checkpoint file.   
        freeze (bool): Whether to set the model to eval mode and freeze its parameters after loading.
    Returns:
        object: An instance of the specified class initialized with the given arguments and loaded with the checkpoint state.
        
    """
    model = instantiate_from_config(config)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            
            state_dict = checkpoint.get("state_dict", checkpoint)  # Handle both cases where state_dict is nested or not
            model.load_state_dict(state_dict,strict=False)  # Use strict=False to allow for missing keys if the checkpoint has extra keys
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from '{ckpt_path}': {e}")
        
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        # Monkey patch the train method to prevent accidental unfreezing
        model.train = lambda mode=True: model
    return model
        
        
def load_from_file(config_path: str, ckpt_path: str, freeze: bool = True):
    """
    Initialize an object from a configuration file and load its state from a checkpoint.

    Args:
        config_path (str): Path to the YAML configuration file.
        ckpt_path (str): Path to the model checkpoint file.   
        freeze (bool): Whether to set the model to eval mode and freeze its parameters after loading.
    Returns:
        object: An instance of the specified class initialized with the given arguments and loaded with the checkpoint state.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_config = config.get("model", config)  # Handle both cases where model config is nested or not
    return load_from_config(model_config, ckpt_path, freeze)

