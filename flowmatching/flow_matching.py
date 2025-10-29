"""
Flow Matching implementation.
Based on "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class FlowMatching:
    """
    Flow Matching for continuous normalizing flows.
    
    Flow Matching learns a vector field that transports samples from
    a simple source distribution (Gaussian noise) to the target data distribution.
    
    Key differences from Diffusion:
    - Continuous time formulation (no discrete timesteps)
    - Direct vector field prediction instead of noise prediction
    - Simpler training objective (straight-line paths)
    - Faster sampling with ODE integration
    """
    
    def __init__(
        self,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize Flow Matching.
        
        Args:
            sigma_min: Minimum noise level (usually 0.0 for Flow Matching)
            sigma_max: Maximum noise level (usually 1.0)
            device: Device to run on
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
        
    def compute_flow_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cfg_prob: float = 0.1
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss.
        
        The training objective is to match the conditional vector field:
        v_t(x_t | x_0) = (x_1 - x_0) / (1 - t)
        
        where x_t = (1-t) * x_0 + t * x_1, and x_1 ~ N(0, I)
        
        Args:
            model: Neural network that predicts the vector field
            x0: Data samples [batch, channels, height, width]
            cond: Conditioning information [batch, cond_dim]
            cfg_prob: Probability of unconditional training for CFG
            
        Returns:
            Flow matching loss (MSE between predicted and true vector field)
        """
        batch_size = x0.shape[0]
        
        # Sample random time t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=self.device)
        
        # Sample target noise x1 ~ N(0, I)
        x1 = torch.randn_like(x0)
        
        # Compute interpolated point on straight line: x_t = (1-t) * x0 + t * x1
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # True vector field: v_t = x1 - x0
        # This is the velocity that moves along the straight line from x0 to x1
        v_true = x1 - x0
        
        # Classifier-free guidance: randomly drop conditioning
        if cond is not None and cfg_prob > 0:
            # Create mask for unconditional training
            uncond_mask = torch.rand(batch_size, device=self.device) < cfg_prob
            cond_input = cond.clone()
            cond_input[uncond_mask] = 0.0  # Zero out conditioning
        else:
            cond_input = cond
            
        # Predict vector field
        v_pred = model(x_t, t, cond_input)
        
        # Flow matching loss: MSE between predicted and true vector field
        loss = nn.functional.mse_loss(v_pred, v_true)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        method: str = 'euler',
        progress: bool = True
    ) -> torch.Tensor:
        """
        Sample using ODE integration.
        
        Integrates the learned vector field from t=1 (noise) to t=0 (data):
        dx/dt = v_t(x_t)
        
        Args:
            model: Trained model
            shape: Output shape (batch, channels, height, width)
            cond: Conditioning information
            num_steps: Number of integration steps
            cfg_scale: Classifier-free guidance scale
            method: Integration method ('euler' or 'heun')
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # Start from pure noise at t=1
        x = torch.randn(shape, device=device)
        
        # Time steps from 1 to 0 (reverse direction)
        times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt = 1.0 / num_steps
        
        if progress:
            from tqdm import tqdm
            iterator = tqdm(range(num_steps), desc="Flow Sampling")
        else:
            iterator = range(num_steps)
        
        for i in iterator:
            t_curr = times[i]
            t_batch = torch.full((batch_size,), t_curr, device=device)
            
            # Predict vector field with CFG
            if cond is not None and cfg_scale > 1.0:
                # Conditional prediction
                v_cond = model(x, t_batch, cond)
                
                # Unconditional prediction
                v_uncond = model(x, t_batch, torch.zeros_like(cond))
                
                # CFG combination
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # Standard prediction
                v = model(x, t_batch, cond)
            
            # Integration step
            if method == 'euler':
                # Simple Euler method: x_{t-dt} = x_t - dt * v_t
                x = x - dt * v
                
            elif method == 'heun':
                # Heun's method (2nd order)
                x_temp = x - dt * v
                
                if i < num_steps - 1:
                    t_next = times[i + 1]
                    t_next_batch = torch.full((batch_size,), t_next, device=device)
                    
                    if cond is not None and cfg_scale > 1.0:
                        v_next_cond = model(x_temp, t_next_batch, cond)
                        v_next_uncond = model(x_temp, t_next_batch, torch.zeros_like(cond))
                        v_next = v_next_uncond + cfg_scale * (v_next_cond - v_next_uncond)
                    else:
                        v_next = model(x_temp, t_next_batch, cond)
                    
                    x = x - dt * 0.5 * (v + v_next)
                else:
                    x = x_temp
            
            else:
                raise ValueError(f"Unknown integration method: {method}")
                
        return x
    
    def sample_ddim_style(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        DDIM-style sampling with optional stochasticity.
        
        Args:
            model: Trained model
            shape: Output shape
            cond: Conditioning
            num_steps: Number of steps
            cfg_scale: CFG scale
            eta: Stochasticity (0 = deterministic, 1 = stochastic)
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Time steps
        times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        if progress:
            from tqdm import tqdm
            iterator = tqdm(range(num_steps), desc="DDIM Sampling")
        else:
            iterator = range(num_steps)
        
        for i in iterator:
            t_curr = times[i]
            t_batch = torch.full((batch_size,), t_curr, device=device)
            
            # Predict vector field
            if cond is not None and cfg_scale > 1.0:
                v_cond = model(x, t_batch, cond)
                v_uncond = model(x, t_batch, torch.zeros_like(cond))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t_batch, cond)
            
            # DDIM step
            if i < num_steps - 1:
                dt = times[i] - times[i + 1]
                
                # Deterministic part
                x = x - dt * v
                
                # Stochastic part (optional)
                if eta > 0:
                    noise = torch.randn_like(x)
                    x = x + eta * torch.sqrt(dt) * noise
            else:
                # Final step
                dt = times[i]
                x = x - dt * v
                
        return x





