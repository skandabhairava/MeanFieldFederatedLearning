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
    return torch.cat([sd[k].view(-1) for k in sorted(sd.keys()) if 'running_' not in k or "num_batches_tracked" not in k])
