import os
import random
import logging as log

import torch
import numpy as np

def set_log_level(level: int):
    logger = log.getLogger()
    logger.setLevel(level)

def set_all_seeds(seed: int = 42) -> None:    
    # Basic Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # cuDNN settings for reproducibility
        torch.backends.cudnn.deterministic = True
        
        torch.backends.cudnn.benchmark = False
    
    try:
        # Some scipy functions use random numbers
        import scipy
        # Scipy doesn't have a global seed function, 
        # but we can seed numpy which scipy uses
        # Note: scipy.stats uses its own RNG in newer versions
        from scipy import stats
        if hasattr(stats, 'rng_global'):
            stats.rng_global = np.random.default_rng(seed) # pyright: ignore[reportAttributeAccessIssue]
    except ImportError:
        pass
    
    # Set Python hash seed for dictionary ordering (if needed)
    os.environ['PYTHONHASHSEED'] = str(seed)