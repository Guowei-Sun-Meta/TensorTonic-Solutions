import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    ap = np.asarray(a, dtype=float)
    bp = np.asarray(b, dtype=float)
    apn = np.linalg.norm(ap)
    bpn = np.linalg.norm(bp)
    if apn == 0 or bpn == 0:
        return 0.0
    # Write code here
    return float((
        np.dot(ap,bp)/(apn*bpn)
    ))