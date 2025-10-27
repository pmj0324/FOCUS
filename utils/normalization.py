"""
Normalization utilities.
"""
import numpy as np
import torch


def denormalize_maps(maps_normalized, stats_path):
    """
    Denormalize normalized maps to original scale
    
    Args:
        maps_normalized: (N, H, W) or (N, 1, H, W) normalized maps
        stats_path: Path to normalization_stats.npy file
        
    Returns:
        maps_original: original scale maps
    """
    stats = np.load(stats_path, allow_pickle=True).item()
    maps_min = stats['maps_min']
    maps_max = stats['maps_max']
    
    # Remove channel dimension if present
    if maps_normalized.ndim == 4:
        maps_normalized = maps_normalized[:, 0]
    
    # Denormalize from [-1, 1]
    maps_log = (maps_normalized + 1) / 2 * (maps_max - maps_min) + maps_min
    
    # Reverse log scale
    maps_original = 10 ** maps_log - 1e-10
    
    return maps_original


def denormalize_params(params_normalized, stats_path):
    """
    Denormalize normalized params to original scale
    
    Args:
        params_normalized: (N, 6) normalized parameters
        stats_path: Path to normalization_stats.npy file
        
    Returns:
        params_original: original scale parameters
    """
    stats = np.load(stats_path, allow_pickle=True).item()
    params_min = stats['params_min']
    params_max = stats['params_max']
    
    # Denormalize from [0, 1]
    params_original = params_normalized * (params_max - params_min) + params_min
    
    return params_original

