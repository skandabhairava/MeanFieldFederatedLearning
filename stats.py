import torch
import numpy as np

from models import StateDict

def flatten(sd: StateDict):
    return torch.cat([sd[k].view(-1) for k in sorted(sd.keys())])

def mean_var(updates):
    mat = torch.stack(updates).numpy()
    return {
        "mean_norm": float(np.linalg.norm(mat.mean(0))),
        "var": float(mat.var()),
    }

def cosine(updates) -> np.ndarray:
    mat = torch.stack(updates)
    mat = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    return (mat @ mat.T).numpy()
