"""
DDPM diffusion process and DDIM sampling.
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .schedules import LinearSchedule


class GaussianDiffusion:
    """
    DDPM-based diffusion process
    """
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule='linear',
        device='cuda'
    ):
        self.timesteps = timesteps
        self.device = device
        
        # Choose noise schedule
        if schedule == 'linear':
            self.schedule = LinearSchedule(timesteps, beta_start, beta_end, device)
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented")
        
        # Pre-compute useful values
        self.betas = self.schedule.get_betas()
        self.alphas = self.schedule.get_alphas()
        self.alphas_cumprod = self.schedule.get_alphas_cumprod()
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: x_0 -> x_t
        
        Args:
            x_start: (B, C, H, W) clean image
            t: (B,) timestep
            noise: (B, C, H, W) noise (optional)
            
        Returns:
            x_t: noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, t, cond, noise=None):
        """
        Training loss computation
        
        Args:
            model: UNet model
            x_start: (B, C, H, W) clean image
            t: (B,) timestep
            cond: (B, param_dim) conditions
            noise: (B, C, H, W) noise (optional)
            
        Returns:
            loss: MSE loss between predicted and true noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, t, cond)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, cond, cfg_scale=1.0):
        """
        Single reverse diffusion step: x_t -> x_{t-1}
        
        Args:
            model: UNet model
            x: (B, C, H, W) noisy image at timestep t
            t: (B,) timestep
            cond: (B, param_dim) conditions
            cfg_scale: Classifier-Free Guidance scale
            
        Returns:
            x_{t-1}: denoised image at timestep t-1
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        
        # Classifier-Free Guidance
        if cfg_scale > 1.0:
            # Conditional prediction
            noise_pred_cond = model(x, t, cond)
            
            # Unconditional prediction (zero condition)
            noise_pred_uncond = model(x, t, torch.zeros_like(cond))
            
            # CFG
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model(x, t, cond)
        
        # Mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, cond, cfg_scale=1.0, progress=True):
        """
        Complete reverse diffusion: x_T -> x_0 (DDPM sampling)
        
        Args:
            model: UNet model
            shape: (B, C, H, W) output shape
            cond: (B, param_dim) conditions
            cfg_scale: CFG scale
            progress: show progress bar
            
        Returns:
            x_0: generated samples
        """
        device = self.device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        timesteps = reversed(range(0, self.timesteps))
        
        if progress:
            timesteps = tqdm(timesteps, desc='Sampling')
        
        for i in timesteps:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond, cfg_scale)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, ddim_timesteps=50, eta=0.0, cfg_scale=1.0, progress=True):
        """
        DDIM sampling (faster than DDPM)
        
        Args:
            model: UNet model
            shape: (B, C, H, W) output shape
            cond: (B, param_dim) conditions
            ddim_timesteps: number of sampling steps
            eta: stochasticity parameter (0 = deterministic)
            cfg_scale: CFG scale
            progress: show progress bar
            
        Returns:
            x_0: generated samples
        """
        device = self.device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create DDIM timestep sequence
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        timesteps = reversed(ddim_timestep_seq)
        
        if progress:
            timesteps = tqdm(list(timesteps), desc='DDIM Sampling')
        
        for i, t_idx in enumerate(timesteps):
            t = torch.full((b,), t_idx, device=device, dtype=torch.long)
            prev_t = torch.full((b,), ddim_timestep_prev_seq[len(ddim_timestep_seq) - i - 1], 
                               device=device, dtype=torch.long)
            
            # CFG
            if cfg_scale > 1.0:
                noise_pred_cond = model(x, t, cond)
                noise_pred_uncond = model(x, t, torch.zeros_like(cond))
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(x, t, cond)
            
            # DDIM update
            alpha_prod_t = self.alphas_cumprod[t][:, None, None, None]
            alpha_prod_t_prev = self.alphas_cumprod[prev_t][:, None, None, None]
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prod_t_prev - eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) * noise_pred
            
            # Add noise
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt
            
            if eta > 0:
                variance = eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                x = x + torch.sqrt(variance) * noise
        
        return x
    
    def check_gaussian_distribution(self, model, x_start, cond, num_steps=10):
        """
        Check if forward diffusion converges to Gaussian
        
        Args:
            model: UNet model (not used, for consistency)
            x_start: (B, C, H, W) clean image
            cond: (B, param_dim) conditions
            num_steps: number of timesteps to check
            
        Returns:
            stats: list of (timestep, mean, std) tuples
        """
        stats = []
        
        step_size = self.timesteps // num_steps
        
        for i in range(num_steps):
            t_val = min(i * step_size, self.timesteps - 1)
            t = torch.full((x_start.shape[0],), t_val, device=self.device, dtype=torch.long)
            
            # Forward diffusion
            x_noisy = self.q_sample(x_start, t)
            
            # Statistics
            mean = x_noisy.mean().item()
            std = x_noisy.std().item()
            
            stats.append((t_val, mean, std))
            
        return stats

