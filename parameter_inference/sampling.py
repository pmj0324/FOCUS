"""
Parameter sampling utilities.
"""
import torch
import numpy as np
from pathlib import Path


class ParameterInference:
    """Parameter inference utilities"""
    
    def __init__(self, model, diffusion, device='cuda'):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        
    @torch.no_grad()
    def infer_from_observation(self, observation, num_samples=100):
        """
        Infer parameters from observation (placeholder for future implementation)
        
        Args:
            observation: observed dark matter map
            num_samples: number of samples to generate
            
        Returns:
            inferred_params: inferred parameters
        """
        raise NotImplementedError("Parameter inference not yet implemented")
    
    @torch.no_grad()
    def sample_prior(self, num_samples=100):
        """
        Sample from prior distribution of parameters
        
        Args:
            num_samples: number of samples
            
        Returns:
            params: sampled parameters
        """
        # For now, sample from uniform distribution in normalized space
        params = torch.rand(num_samples, 6).to(self.device)
        return params

