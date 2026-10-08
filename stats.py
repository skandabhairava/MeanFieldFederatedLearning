from collections import OrderedDict

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import StateDict

from typing import Iterable

def avg(iterator: Iterable[float|int]):
    t = 1e-12
    sum = 0
    for item in iterator:
        sum += item
        t += 1

    return sum/t

# avg(i for i in range(10))

def _flat_keys(sd):
    return [k for k in sorted(sd.keys())
            if 'running_' not in k and 'num_batches_tracked' not in k]

def flatten(sd: 'StateDict'):
    return torch.cat([sd[k].reshape(-1) for k in _flat_keys(sd)])

def unflatten(vec: torch.Tensor, template: 'StateDict') -> 'StateDict':
    out, i = OrderedDict(), 0
    for k in _flat_keys(template):
        n = template[k].numel()
        out[k] = vec[i:i + n].reshape_as(template[k]).to(template[k].dtype)
        i += n
    assert i == vec.numel(), "flat vector size doesn't match template"
    return out