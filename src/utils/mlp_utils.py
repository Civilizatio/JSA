import torch.nn as nn

# def get_act_func_by_name(name):
#     """
#     Get activation function by name.
#     Args:
#         name (str): Name of the activation function (e.g., "relu", "tanh").
#     Returns:
#         nn.Module: Corresponding activation function.
#     """
#     try:
#         return getattr(nn, name.capitalize())()
#     except AttributeError:
#         raise ValueError(f"Unsupported activation function: {name}")
    
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
    }

    if name not in mapping:
        raise ValueError(f"Unsupported activation function: {name}")

    return mapping[name]()


def build_mlp(input_dim, layers, output_dim, activation, final_activation=None):
    """
    Build a multi-layer perceptron (MLP).
    Args:
        input_dim (int): Input dimension.
        layers (list of int): List of hidden layer sizes.
        output_dim (int): Output dimension.
        activation (str): Name of the activation function.
        final_activation (str, optional): Name of the final activation function. Defaults to None.
            - If "sigmoid", output will be in [0, 1]
            - If "tanh", output will be in [-1, 1]
            - If None, no activation on final layer.
    Returns:
        nn.Sequential: The constructed MLP.
    """
    modules = []
    in_dim = input_dim
    for layer_dim in layers:
        modules.append(nn.Linear(in_dim, layer_dim))
        act_func = get_act_func_by_name(activation)
        modules.append(act_func)
        in_dim = layer_dim
    # Last layer without activation
    if final_activation is not None:
        modules.append(nn.Linear(in_dim, output_dim))
        final_act_func = get_act_func_by_name(final_activation)
        modules.append(final_act_func)
    else:
        modules.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*modules)