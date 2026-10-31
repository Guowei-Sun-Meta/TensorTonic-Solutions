import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    x = np.asarray(X, dtype = float)# Write code here
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    d = xmax-xmin
    s = np.where(d > eps, d, 1.0)
    return (x-xmin) / s