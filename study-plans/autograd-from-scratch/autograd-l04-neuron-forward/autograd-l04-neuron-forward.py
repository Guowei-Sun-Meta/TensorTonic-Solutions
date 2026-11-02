import torch

def neuron_forward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: preactivation and tanh output.
    """
    # return (torch.sum(inputs*weights) + bias ,torch.tanh( torch.sum(inputs * weights) + bias ))
    return (inputs @ weights.T + bias, torch.tanh(inputs @ weights.T + bias))
