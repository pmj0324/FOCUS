"""
Parameter Inference using Flow Matching Model

This script implements parameter inference for cosmological field generation
using a trained Flow Matching model. Given observed data (2D dark matter maps),
it estimates the cosmological parameters through likelihood-based inference.

Methods implemented:
1. Negative Log-Likelihood (NLL) estimation via ODE probability flow
2. Maximum Likelihood Estimation (MLE) via gradient-based optimization
3. Bayesian inference via MCMC sampling
4. Grid search for approximate posterior

Author: Flow Matching Parameter Inference
Date: 2025-10-29
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from typing import Optional, Tuple, Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flowmatching.flow_matching import FlowMatching
from flowmatching.flow_model import FlowUNet
from utils.visualization import visualize_samples


class FlowMatchingLikelihood:
    """
    Compute likelihood for Flow Matching models using probability flow ODE.
    
    For flow matching, the likelihood can be computed by integrating the
    probability flow ODE along with the trace of the Jacobian:
    
    log p(x_0) = log p(x_1) - ∫_0^1 ∇·v(x_t, t) dt
    
    where v is the learned vector field.
    """
    
    def __init__(self, model: nn.Module, flow_matching: FlowMatching, device: str = 'cuda'):
        """
        Initialize likelihood estimator.
        
        Args:
            model: Trained FlowUNet model
            flow_matching: FlowMatching instance
            device: Device to run on
        """
        self.model = model
        self.flow_matching = flow_matching
        self.device = device
        self.model.eval()
    
    @torch.enable_grad()
    def compute_divergence(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of the vector field using Hutchinson's trace estimator.
        
        ∇·v(x_t, t) ≈ ε^T (∂v/∂x_t) ε, where ε ~ N(0, I)
        
        Args:
            x_t: Current state [batch, channels, height, width]
            t: Time values [batch]
            cond: Conditioning parameters [batch, cond_dim]
            
        Returns:
            Divergence estimates [batch]
        """
        x_t.requires_grad_(True)
        batch_size = x_t.shape[0]
        
        # Random vector for Hutchinson estimator
        epsilon = torch.randn_like(x_t)
        
        # Predict vector field
        with torch.enable_grad():
            v = self.model(x_t, t, cond)
            
            # Compute v^T ε
            v_eps = (v * epsilon).sum()
            
            # Compute gradient: ∂(v^T ε)/∂x_t
            grad_outputs = torch.ones_like(v_eps)
            grad_v_eps = torch.autograd.grad(
                v_eps, x_t, 
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Divergence: ε^T (∂v/∂x_t) ε
            divergence = (grad_v_eps * epsilon).view(batch_size, -1).sum(dim=1)
        
        x_t.requires_grad_(False)
        return divergence
    
    @torch.no_grad()
    def compute_nll_approximate(
        self, 
        x_0: torch.Tensor, 
        cond: torch.Tensor,
        num_steps: int = 50,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Compute approximate negative log-likelihood using simplified flow.
        
        This uses a simpler approximation: measuring how well the model
        can reconstruct the data from noise.
        
        Args:
            x_0: Observed data [batch, channels, height, width]
            cond: Conditioning parameters [batch, cond_dim]
            num_steps: Number of ODE steps
            method: Integration method
            
        Returns:
            Approximate NLL [batch]
        """
        batch_size = x_0.shape[0]
        
        # Forward: add noise to data to get x_t at different times
        # Compute reconstruction error as proxy for likelihood
        
        # Sample multiple time points
        t_values = torch.linspace(0.1, 0.9, 5, device=self.device)
        total_error = 0.0
        
        for t_val in t_values:
            t = torch.full((batch_size,), t_val, device=self.device)
            
            # Add noise: x_t = (1-t) * x_0 + t * x_1
            x_1 = torch.randn_like(x_0)
            t_expanded = t.view(-1, 1, 1, 1)
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            
            # Predict vector field
            v_pred = self.model(x_t, t, cond)
            
            # True vector field
            v_true = x_1 - x_0
            
            # Compute error
            error = ((v_pred - v_true) ** 2).view(batch_size, -1).mean(dim=1)
            total_error = total_error + error
        
        # Average over time points
        nll = total_error / len(t_values)
        
        return nll
    
    def compute_reconstruction_error(
        self,
        x_0: torch.Tensor,
        cond: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Compute reconstruction error: generate from cond and compare to x_0.
        
        This is a practical proxy for likelihood.
        This version computes gradients w.r.t. cond for parameter optimization.
        
        Args:
            x_0: Observed data [batch, channels, height, width]
            cond: Conditioning parameters [batch, cond_dim]
            num_samples: Number of samples to generate per observation
            
        Returns:
            Reconstruction error [batch]
        """
        batch_size = x_0.shape[0]
        
        # For gradient-based optimization, we use a differentiable proxy:
        # Instead of full sampling (which is not differentiable), we evaluate
        # the vector field prediction error at random times
        
        # Sample multiple time points
        num_time_samples = 5
        t_values = torch.linspace(0.1, 0.9, num_time_samples, device=self.device)
        total_error = 0.0
        
        for t_val in t_values:
            t = torch.full((batch_size,), t_val, device=self.device)
            
            # Add noise: x_t = (1-t) * x_0 + t * x_1
            with torch.no_grad():
                x_1 = torch.randn_like(x_0)
            t_expanded = t.view(-1, 1, 1, 1)
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            
            # Predict vector field (this is differentiable w.r.t. cond)
            v_pred = self.model(x_t, t, cond)
            
            # True vector field
            v_true = x_1 - x_0
            
            # Compute error
            error = ((v_pred - v_true) ** 2).view(batch_size, -1).mean(dim=1)
            total_error = total_error + error
        
        # Average over time points
        return total_error / num_time_samples


class ParameterInference:
    """
    Parameter inference for Flow Matching models.
    
    Given observed data x_obs, estimate parameters θ using various methods:
    1. Maximum Likelihood Estimation (MLE)
    2. Maximum A Posteriori (MAP) with prior
    3. MCMC sampling for full posterior
    4. Grid search
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = 'cuda'
    ):
        """
        Initialize parameter inference.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to config file
            device: Device to run on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = self._load_model()
        
        # Initialize Flow Matching
        self.flow_matching = FlowMatching(
            sigma_min=self.config['flow_matching']['sigma_min'],
            sigma_max=self.config['flow_matching']['sigma_max'],
            device=device
        )
        
        # Initialize likelihood estimator
        self.likelihood = FlowMatchingLikelihood(
            self.model,
            self.flow_matching,
            device
        )
        
        # Load normalization stats
        self._load_normalization_stats()
        
        print("✓ Parameter inference initialized")
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        # Create model
        model_config = self.config['model']
        model = FlowUNet(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            cond_dim=model_config['cond_dim'],
            base_channels=model_config['base_channels'],
            channel_mults=tuple(model_config['channel_mults']),
            time_dim=model_config.get('time_dim', 256)
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded from {self.checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
        
        return model
    
    def _load_normalization_stats(self):
        """Load normalization statistics."""
        data_dir = Path(self.config['data']['maps_path']).parent
        stats_path = data_dir / 'normalization_stats.npy'
        
        if stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True).item()
            self.param_min = torch.tensor(stats['params_min'], device=self.device, dtype=torch.float32)
            self.param_max = torch.tensor(stats['params_max'], device=self.device, dtype=torch.float32)
            print(f"✓ Normalization stats loaded")
        else:
            print("⚠ Normalization stats not found, using default range [0, 1]")
            self.param_min = torch.zeros(6, device=self.device)
            self.param_max = torch.ones(6, device=self.device)
    
    def normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize parameters to [0, 1]."""
        return (params - self.param_min) / (self.param_max - self.param_min)
    
    def denormalize_params(self, params_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize parameters from [0, 1]."""
        return params_norm * (self.param_max - self.param_min) + self.param_min
    
    def infer_mle(
        self,
        x_obs: torch.Tensor,
        init_params: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        lr: float = 0.01,
        method: str = 'reconstruction'
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Maximum Likelihood Estimation using gradient-based optimization.
        
        Args:
            x_obs: Observed data [1, channels, height, width]
            init_params: Initial parameter guess [1, 6] (normalized)
            num_iterations: Number of optimization steps
            lr: Learning rate
            method: 'reconstruction' or 'nll'
            
        Returns:
            Estimated parameters (denormalized), loss history
        """
        print("\n" + "="*60)
        print("Maximum Likelihood Estimation")
        print("="*60)
        
        # Initialize parameters
        if init_params is None:
            # Random initialization in normalized space [0, 1]
            params = torch.rand(1, 6, device=self.device, requires_grad=True)
        else:
            params = init_params.clone().detach().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([params], lr=lr)
        
        # Optimization loop
        losses = []
        pbar = tqdm(range(num_iterations), desc="MLE Optimization")
        
        for i in pbar:
            optimizer.zero_grad()
            
            # Ensure parameters stay in [0, 1]
            with torch.no_grad():
                params.data.clamp_(0, 1)
            
            # Compute negative log-likelihood
            if method == 'reconstruction':
                nll = self.likelihood.compute_reconstruction_error(
                    x_obs, params, num_samples=1
                )
            else:
                nll = self.likelihood.compute_nll_approximate(
                    x_obs, params, num_steps=20
                )
            
            loss = nll.mean()
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Record
            losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Denormalize final parameters
        params_final = self.denormalize_params(params.detach())
        
        print(f"\n✓ MLE completed")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Estimated parameters (denormalized):")
        print(f"    {params_final.cpu().numpy()[0]}")
        
        return params_final, losses
    
    @torch.no_grad()
    def infer_grid_search(
        self,
        x_obs: torch.Tensor,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        grid_points: int = 10
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Grid search over parameter space.
        
        Args:
            x_obs: Observed data [1, channels, height, width]
            param_ranges: Dictionary of parameter ranges (denormalized)
            grid_points: Number of points per parameter
            
        Returns:
            Best parameters (denormalized), grid of likelihoods
        """
        print("\n" + "="*60)
        print("Grid Search Parameter Inference")
        print("="*60)
        
        # For simplicity, search over 2 parameters at a time
        # You can extend this to more dimensions
        param_idx = [0, 1]  # Search over first 2 parameters
        
        if param_ranges is None:
            # Use full normalized range
            range_0 = torch.linspace(0, 1, grid_points, device=self.device)
            range_1 = torch.linspace(0, 1, grid_points, device=self.device)
        else:
            # Use provided ranges (denormalized)
            # Not implemented yet - use normalized range
            range_0 = torch.linspace(0, 1, grid_points, device=self.device)
            range_1 = torch.linspace(0, 1, grid_points, device=self.device)
        
        # Create grid
        grid_0, grid_1 = torch.meshgrid(range_0, range_1, indexing='ij')
        
        # Evaluate likelihood on grid
        likelihoods = np.zeros((grid_points, grid_points))
        
        # Use mean parameters for other dimensions
        mean_params = torch.ones(1, 6, device=self.device) * 0.5
        
        print(f"Evaluating {grid_points}x{grid_points} grid...")
        for i in tqdm(range(grid_points)):
            for j in range(grid_points):
                params = mean_params.clone()
                params[0, param_idx[0]] = grid_0[i, j]
                params[0, param_idx[1]] = grid_1[i, j]
                
                # Compute likelihood
                nll = self.likelihood.compute_reconstruction_error(
                    x_obs, params, num_samples=1
                )
                likelihoods[i, j] = -nll.item()  # Convert to likelihood
        
        # Find best parameters
        best_idx = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
        best_params_norm = mean_params.clone()
        best_params_norm[0, param_idx[0]] = grid_0[best_idx]
        best_params_norm[0, param_idx[1]] = grid_1[best_idx]
        
        best_params = self.denormalize_params(best_params_norm)
        
        print(f"\n✓ Grid search completed")
        print(f"  Best likelihood: {likelihoods[best_idx]:.6f}")
        print(f"  Best parameters (denormalized):")
        print(f"    {best_params.cpu().numpy()[0]}")
        
        return best_params, likelihoods
    
    @torch.no_grad()
    def infer_mcmc(
        self,
        x_obs: torch.Tensor,
        num_samples: int = 1000,
        burn_in: int = 100,
        proposal_std: float = 0.05
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Metropolis-Hastings MCMC sampling.
        
        Args:
            x_obs: Observed data [1, channels, height, width]
            num_samples: Number of MCMC samples
            burn_in: Burn-in period
            proposal_std: Standard deviation of proposal distribution
            
        Returns:
            Posterior samples (denormalized), acceptance rate
        """
        print("\n" + "="*60)
        print("MCMC Sampling")
        print("="*60)
        
        # Initialize at random
        current_params = torch.rand(1, 6, device=self.device)
        
        # Compute initial likelihood
        current_nll = self.likelihood.compute_reconstruction_error(
            x_obs, current_params, num_samples=1
        ).item()
        
        # Storage
        samples = []
        accepted = 0
        
        # MCMC loop
        pbar = tqdm(range(num_samples + burn_in), desc="MCMC Sampling")
        for i in pbar:
            # Propose new parameters
            proposal = current_params + torch.randn_like(current_params) * proposal_std
            proposal.clamp_(0, 1)  # Stay in bounds
            
            # Compute likelihood
            proposal_nll = self.likelihood.compute_reconstruction_error(
                x_obs, proposal, num_samples=1
            ).item()
            
            # Acceptance ratio (likelihood ratio)
            # p(proposal) / p(current) = exp(-nll_proposal) / exp(-nll_current)
            #                          = exp(nll_current - nll_proposal)
            log_alpha = current_nll - proposal_nll
            
            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                current_params = proposal
                current_nll = proposal_nll
                accepted += 1
            
            # Store sample (after burn-in)
            if i >= burn_in:
                samples.append(current_params.cpu().numpy()[0])
            
            # Update progress
            if i > 0:
                acc_rate = accepted / (i + 1)
                pbar.set_postfix({'acceptance': f'{acc_rate:.3f}', 'nll': f'{current_nll:.4f}'})
        
        # Convert to array
        samples = np.array(samples)
        
        # Denormalize
        samples_denorm = self.denormalize_params(
            torch.tensor(samples, device=self.device)
        ).cpu().numpy()
        
        acceptance_rate = accepted / (num_samples + burn_in)
        
        print(f"\n✓ MCMC completed")
        print(f"  Acceptance rate: {acceptance_rate:.3f}")
        print(f"  Posterior mean (denormalized):")
        print(f"    {samples_denorm.mean(axis=0)}")
        print(f"  Posterior std (denormalized):")
        print(f"    {samples_denorm.std(axis=0)}")
        
        return samples_denorm, acceptance_rate
    
    def visualize_results(
        self,
        x_obs: torch.Tensor,
        params_estimated: torch.Tensor,
        params_true: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize inference results.
        
        Args:
            x_obs: Observed data
            params_estimated: Estimated parameters
            params_true: True parameters (if known)
            save_path: Path to save figure
        """
        # Generate samples with estimated parameters
        params_norm = self.normalize_params(params_estimated)
        
        x_gen = self.flow_matching.sample(
            self.model,
            shape=(4, 1, 256, 256),
            cond=params_norm.repeat(4, 1),
            num_steps=50,
            cfg_scale=2.0,
            method='euler',
            progress=True
        )
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Observed
        im0 = axes[0, 0].imshow(x_obs[0, 0].cpu().numpy(), cmap='viridis')
        axes[0, 0].set_title("Observed Data", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        
        # Generated samples
        for i in range(4):
            row = (i + 1) // 3
            col = (i + 1) % 3
            if row < 2 and col < 3:
                im = axes[row, col].imshow(x_gen[i, 0].cpu().numpy(), cmap='viridis')
                axes[row, col].set_title(f"Generated Sample {i+1}", fontsize=12)
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        # Remove extra subplot
        if len(axes.flatten()) > 5:
            fig.delaxes(axes[1, 2])
        
        # Add parameter text
        param_text = "Estimated Parameters:\n"
        param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w']
        for i, (name, val) in enumerate(zip(param_names, params_estimated[0].cpu().numpy())):
            param_text += f"  {name}: {val:.4f}\n"
        
        if params_true is not None:
            param_text += "\nTrue Parameters:\n"
            for i, (name, val) in enumerate(zip(param_names, params_true[0].cpu().numpy())):
                param_text += f"  {name}: {val:.4f}\n"
        
        fig.text(0.68, 0.25, param_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Results saved to {save_path}")
        
        plt.show()


def main():
    """Main function for parameter inference."""
    # Paths
    exp_dir = Path(__file__).parent
    checkpoint_path = exp_dir / 'checkpoints' / 'checkpoint_best.pt'
    config_path = exp_dir / 'config.yaml'
    
    # Initialize
    inferencer = ParameterInference(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load test data (replace with your observed data)
    data_dir = Path(inferencer.config['data']['maps_path']).parent
    maps = np.load(data_dir / 'maps_normalized.npy')
    params = np.load(data_dir / 'params_normalized.npy')
    
    # Use a random test sample
    test_idx = np.random.randint(0, len(maps))
    x_obs = torch.tensor(maps[test_idx:test_idx+1], device=inferencer.device, dtype=torch.float32)
    
    # Ensure x_obs has channel dimension [batch, channels, height, width]
    if x_obs.ndim == 3:
        x_obs = x_obs.unsqueeze(1)
    
    params_true = torch.tensor(params[test_idx:test_idx+1], device=inferencer.device, dtype=torch.float32)
    
    print(f"\nTest sample index: {test_idx}")
    print(f"True parameters (normalized): {params_true.cpu().numpy()[0]}")
    params_true_denorm = inferencer.denormalize_params(params_true)
    print(f"True parameters (denormalized): {params_true_denorm.cpu().numpy()[0]}")
    
    # Method 1: MLE
    params_mle, losses_mle = inferencer.infer_mle(
        x_obs,
        num_iterations=200,
        lr=0.01,
        method='reconstruction'
    )
    
    # Method 2: Grid Search (2D search over first 2 parameters)
    # params_grid, likelihoods_grid = inferencer.infer_grid_search(
    #     x_obs,
    #     grid_points=20
    # )
    
    # Method 3: MCMC
    # samples_mcmc, acc_rate = inferencer.infer_mcmc(
    #     x_obs,
    #     num_samples=1000,
    #     burn_in=100,
    #     proposal_std=0.05
    # )
    
    # Visualize results
    output_dir = exp_dir / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    inferencer.visualize_results(
        x_obs,
        params_mle,
        params_true_denorm,
        save_path=str(output_dir / 'inference_mle_results.png')
    )
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(losses_mle)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Negative Log-Likelihood', fontsize=12)
    plt.title('MLE Optimization History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'mle_loss_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Parameter Inference Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

