"""
Noise schedules for diffusion models.
"""
import torch
import numpy as np


class NoiseSchedule:
    """Base class for noise schedules."""
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
    def get_betas(self):
        """Compute beta schedule."""
        raise NotImplementedError
        
    def get_alphas(self):
        """Compute alpha schedule."""
        raise NotImplementedError


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__(timesteps, device)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._betas = None
        self._alphas = None
        self._alphas_cumprod = None
        
    def get_betas(self):
        """Linear beta schedule."""
        if self._betas is None:
            self._betas = torch.linspace(
                self.beta_start, self.beta_end, self.timesteps
            ).to(self.device)
        return self._betas
    
    def get_alphas(self):
        """Alpha from beta."""
        if self._alphas is None:
            self._alphas = 1.0 - self.get_betas()
        return self._alphas
    
    def get_alphas_cumprod(self):
        """Cumulative product of alphas."""
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.get_alphas(), dim=0)
        return self._alphas_cumprod


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule (better for image generation)."""
    def __init__(self, timesteps=1000, s=0.008, device='cuda'):
        super().__init__(timesteps, device)
        self.s = s
        self._betas = None
        self._alphas = None
        self._alphas_cumprod = None
        
    def get_betas(self):
        """Cosine beta schedule."""
        if self._betas is None:
            steps = torch.arange(self.timesteps + 1, dtype=torch.float32)
            steps = steps / self.timesteps
            alphas_cumprod = torch.cos((steps + self.s) / (1 + self.s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            
            betas = []
            for i in range(self.timesteps):
                beta = 1 - (alphas_cumprod[i + 1] / alphas_cumprod[i])
                betas.append(beta)
            
            self._betas = torch.tensor(betas).to(self.device)
        return self._betas
    
    def get_alphas(self):
        """Alpha from beta."""
        if self._alphas is None:
            self._alphas = 1.0 - self.get_betas()
        return self._alphas
    
    def get_alphas_cumprod(self):
        """Cumulative product of alphas."""
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.get_alphas(), dim=0)
        return self._alphas_cumprod


class QuadraticSchedule(NoiseSchedule):
    """Quadratic noise schedule."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__(timesteps, device)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._betas = None
        self._alphas = None
        self._alphas_cumprod = None
        
    def get_betas(self):
        """Quadratic beta schedule."""
        if self._betas is None:
            steps = torch.linspace(0, 1, self.timesteps)
            self._betas = torch.linspace(
                self.beta_start ** 0.5, self.beta_end ** 0.5, self.timesteps
            ) ** 2
            self._betas = self._betas.to(self.device)
        return self._betas
    
    def get_alphas(self):
        """Alpha from beta."""
        if self._alphas is None:
            self._alphas = 1.0 - self.get_betas()
        return self._alphas
    
    def get_alphas_cumprod(self):
        """Cumulative product of alphas."""
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.get_alphas(), dim=0)
        return self._alphas_cumprod

