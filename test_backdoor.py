import torch

def dfbscanner_detect(state_dict: dict[str, torch.Tensor], 
                      num_classes: int) -> tuple[float, float]:
    """
    DFBScanner: Detect backdoors via final-layer parameter analysis.
    Works purely on state_dict - no data, no model forward needed.
    
    Args:
        state_dict: PyTorch model state_dict
        num_classes: Number of output classes
        threshold: Anomaly detection threshold (default 2.0)
        
    Returns:
        (is_backdoored, target_label, anomaly_score)
    """
    # Find final layer weight and bias
    final_weight = None
    final_bias = None
    
    for key in sorted(state_dict.keys()):
        if 'weight' in key and 'bn' not in key and 'norm' not in key:
            final_weight = state_dict[key]
        if 'bias' in key and 'bn' not in key and 'norm' not in key:
            final_bias = state_dict[key]
    
    if final_weight is None:
        raise ValueError("No final layer weight found in state_dict")
    
    # DFBScanner indicators (simplified version)
    # Indicator 1: Weight magnitude anomaly per class
    weight_norms = torch.norm(final_weight, dim=1)  # L2 norm per output class
    
    # Indicator 2: Bias anomaly (if bias exists)
    if final_bias is not None:
        bias_scores = torch.abs(final_bias)
        combined_scores = weight_norms + bias_scores
    else:
        combined_scores = weight_norms
    
    # Indicator 3: Weight distribution skewness per class
    weight_skew = torch.zeros(num_classes)
    for i in range(min(num_classes, final_weight.shape[0])):
        class_weights = final_weight[i].flatten()
        # Compute skewness: (mean - median) / std
        mean_w = torch.mean(class_weights)
        median_w = torch.median(class_weights)
        std_w = torch.std(class_weights) + 1e-8
        weight_skew[i] = torch.abs((mean_w - median_w) / std_w)
    
    # Combine indicators
    anomaly_scores = combined_scores[:num_classes] * (1 + weight_skew)
    
    # MAD (Median Absolute Deviation) outlier detection
    median_score = torch.median(anomaly_scores)
    mad = torch.median(torch.abs(anomaly_scores - median_score))
    mad = mad + 1e-8  # Avoid division by zero
    
    anomaly_indices = torch.abs(anomaly_scores - median_score) / mad
    
    # Find most anomalous class
    max_anomaly, max_idx = torch.max(anomaly_indices, dim=0)
    
    return max_anomaly.item(), max_idx.item()

import torch

def spectral_signature_detect(
        state_dict: dict[str, torch.Tensor],
    ) -> float:
    """
    Spectral Signature: Detect backdoors via SVD analysis of weight matrices.
    No data or model execution needed - pure weight-space analysis.
    
    Args:
        state_dict: PyTorch model state_dict
        
    Returns:
        (is_backdoored, max_energy_concentration, layer_scores)
    """
    # layer_scores = []
    max_energy = 0.0
    
    for name, param in state_dict.items():
        if 'weight' not in name or param.ndim < 2:
            continue
            
        # Reshape to 2D matrix
        w = param.detach().cpu()
        if w.ndim > 2:
            w = w.view(w.size(0), -1)
        
        # SVD
        try:
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        except:
            continue
        
        if S.numel() == 0:
            continue
        
        # Spectral metrics
        total_energy = torch.sum(S)
        if total_energy < 1e-8:
            continue
            
        # Energy concentration: fraction in top singular value
        energy_concentration = S[0] / total_energy
        
        # Spectral entropy
        p = S / total_energy
        p = p[p > 0]  # Remove zeros
        spectral_entropy = -torch.sum(p * torch.log(p + 1e-10))
        
        # Kurtosis of singular values
        mean_s = torch.mean(S)
        std_s = torch.std(S) + 1e-8
        kurtosis = torch.mean(((S - mean_s) / std_s) ** 4)
        
        # Combined anomaly score
        score = (energy_concentration * 0.5 + 
                 (1.0 / (spectral_entropy + 1.0)) * 0.3 + 
                 (kurtosis / 10.0) * 0.2)
        
        # layer_scores.append((name, float(energy_concentration.item())))
        
        if energy_concentration > max_energy:
            max_energy = float(energy_concentration.item())
    
    return max_energy