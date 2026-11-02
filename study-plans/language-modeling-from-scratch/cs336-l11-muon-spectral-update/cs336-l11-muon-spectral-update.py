import torch

def muon_spectral_update(
    parameter: torch.Tensor, gradient: torch.Tensor,
    previous_momentum: torch.Tensor, momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum, orthogonalized_update.
    """
    new_momentum = momentum_coefficient * previous_momentum + gradient
    U, S, Vt = torch.linalg.svd(new_momentum.float(), full_matrices = False)
    Q = (U @ Vt).to(parameter.dtype)
    new_parameter = parameter - learning_rate * Q
    return {"new_parameter": new_parameter, "new_momentum": new_momentum, "orthogonalized_update":Q}
