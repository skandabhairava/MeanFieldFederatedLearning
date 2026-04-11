import torch
from typing import Sequence
import models

import logging as log
import stats

def krum_aggregate_adaptive(
    client_states: Sequence[tuple[int, models.StateDict]],
    num_byzantine: int | None = None,
) -> models.StateDict:
    n = len(client_states)
    
    # Handle trivial cases early
    if n == 0:
        raise ValueError("No client states provided")
    if n < 3:
        return coordinate_wise_median(client_states)
    
    # Estimate Byzantine count if not provided
    if num_byzantine is None:
        num_byzantine = (n - 3) // 2   # maximum Krum can tolerate
    else:
        num_byzantine = max(0, num_byzantine)
    
    # Ensure Krum's condition holds; otherwise fallback to median
    if n < 2 * num_byzantine + 3:
        num_byzantine = max(0, (n - 3) // 2)
        if n < 2 * num_byzantine + 3:   # still not satisfied?
            log.info("!!!! Krum condition not held. Reverting back to median 1.")
            return coordinate_wise_median(client_states)
    
    k = n - num_byzantine - 2
    if k <= 0:
        log.info("!!!! Krum condition not held. Reverting back to median 2.")
        return coordinate_wise_median(client_states)
    
    # Flatten parameters into vectors
    vectors = []
    for _, state in client_states:
        flat =  stats.flatten(state) #torch.cat([p.flatten() for p in state.values()])
        vectors.append(flat)
    vectors = torch.stack(vectors)   # shape (n, d)
    
    # Compute pairwise Euclidean distances (efficient, vectorized)
    # Using torch.cdist – much faster than nested loops
    dist_matrix = torch.cdist(vectors, vectors, p=2)  # shape (n, n)
    
    # Compute Krum scores
    scores = []
    for i in range(n):
        # Exclude self (distance 0) and take the k smallest distances
        sorted_dists = torch.sort(dist_matrix[i])[0][1:k+1]
        scores.append(sorted_dists.sum())
    
    best_idx = int(torch.argmin(torch.tensor(scores)).item())
    return client_states[best_idx][1]

def coordinate_wise_median(client_states) -> models.StateDict:
    """Fallback aggregation when Krum assumptions can't be met."""
    states = [state for _, state in client_states]
    agg_state = {}
    for key in states[0].keys():
        stacked = torch.stack([s[key] for s in states])
        median_vals, _ = torch.median(stacked, dim=0)
        agg_state[key] = median_vals
    return agg_state # pyright: ignore[reportReturnType]