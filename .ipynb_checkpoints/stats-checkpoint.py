import torch
import numpy as np

from models import StateDict

from typing import Iterable

def avg(iterator: Iterable[float|int]):
    t = 0
    sum = 0
    for item in iterator:
        sum += item
        t += 1

    return sum/t

# avg(i for i in range(10))

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
