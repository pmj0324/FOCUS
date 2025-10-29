"""
Diffusion model trainer.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from .callbacks import EarlyStopping, ModelCheckpoint, Logger


class DiffusionTrainer:
    """
    Diffusion model trainer class
    """
    def __init__(
        self,
        model,
        diffusion,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-4,
        weight_decay=1e-4,
        cfg_prob=0.1,  # CFG unconditional probability
        output_dir='./outputs',
        optimizer=None,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg_prob = cfg_prob
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Optimizer - use provided or create default
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        
        # Learning rate scheduler - use provided or create default
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-7
            )
        
        # Callbacks
        self.early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        self.checkpoint = ModelCheckpoint(output_dir)
        self.logger = Logger(output_dir)
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, params) in enumerate(pbar):
            images = images.to(self.device)
            params = params.to(self.device)
            
            # Random timesteps
            t = torch.randint(
                0, self.diffusion.timesteps, (images.shape[0],), device=self.device
            ).long()
            
            # Classifier-Free Guidance: randomly drop conditions
            if self.cfg_prob > 0:
                mask = torch.rand(params.shape[0], device=self.device) < self.cfg_prob
                params_masked = params.clone()
                params_masked[mask] = 0  # Zero out conditions
            else:
                params_masked = params
            
            # Compute loss
            loss = self.diffusion.p_losses(self.model, images, t, params_masked)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        for images, params in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            params = params.to(self.device)
            
            # Random timesteps
            t = torch.randint(
                0, self.diffusion.timesteps, (images.shape[0],), device=self.device
            ).long()
            
            # Compute loss (no CFG masking for validation)
            loss = self.diffusion.p_losses(self.model, images, t, params)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    @torch.no_grad()
    def sample_and_save(self, epoch, num_samples=4, cfg_scale=2.0):
        """Generate and save samples with real comparison and power spectrum"""
        self.model.eval()
        
        # Get conditions and real images from validation data
        real_images, params = next(iter(self.val_loader))
        # Ensure we don't exceed the available batch size
        actual_samples = min(num_samples, params.shape[0])
        real_images = real_images[:actual_samples].to(self.device)
        params = params[:actual_samples].to(self.device)
        
        # DDIM sampling - generate 3 samples per condition
        all_generated = []
        for _ in range(3):
            samples = self.diffusion.ddim_sample(
                self.model,
                shape=(actual_samples, 1, 256, 256),
                cond=params,
                ddim_timesteps=50,
                cfg_scale=cfg_scale,
                progress=False
            )
            all_generated.append(samples)
        
        # Import power spectrum utility
        try:
            from utils.power_spectrum import compute_power_spectrum
            has_power_spectrum = True
        except ImportError:
            has_power_spectrum = False
        
        # Visualize: For each sample, show [Real, Gen1, Gen2, Gen3]
        fig = plt.figure(figsize=(16, actual_samples * 4))
        
        for i in range(actual_samples):
            # Real image
            ax = plt.subplot(actual_samples, 4, i * 4 + 1)
            real_img = real_images[i, 0].cpu().numpy()
            ax.imshow(real_img, cmap='viridis')
            ax.axis('off')
            if i == 0:
                ax.set_title('Real', fontsize=12, fontweight='bold')
            
            # Parameter text
            param_vals = params[i].cpu().numpy()
            param_names = ['Ωm', 'Ωb', 'h', 'ns', 'σ8', 'w0']
            param_text = '\n'.join([f'{name}={val:.3f}' for name, val in zip(param_names, param_vals)])
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Generated images
            for j in range(3):
                ax = plt.subplot(actual_samples, 4, i * 4 + j + 2)
                gen_img = all_generated[j][i, 0].cpu().numpy()
                ax.imshow(gen_img, cmap='viridis')
                ax.axis('off')
                if i == 0:
                    ax.set_title(f'Generated {j+1}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir.parent / 'figs' / f'samples_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Power spectrum comparison
        if has_power_spectrum:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(min(4, actual_samples)):
                ax = axes[i]
                
                # Compute power spectra
                real_img = real_images[i, 0].cpu().numpy()
                k_real, P_real = compute_power_spectrum(real_img)
                
                # Plot real
                ax.loglog(k_real, P_real, 'k-', linewidth=2, label='Real', alpha=0.8)
                
                # Plot generated
                colors = ['r', 'g', 'b']
                for j, gen_samples in enumerate(all_generated):
                    gen_img = gen_samples[i, 0].cpu().numpy()
                    k_gen, P_gen = compute_power_spectrum(gen_img)
                    ax.loglog(k_gen, P_gen, f'{colors[j]}--', linewidth=1.5, 
                             label=f'Gen {j+1}', alpha=0.7)
                
                ax.set_xlabel('k', fontsize=10)
                ax.set_ylabel('P(k)', fontsize=10)
                ax.set_title(f'Sample {i+1} Power Spectrum', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir.parent / 'figs' / f'power_spectrum_epoch_{epoch:04d}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def train(self, num_epochs=100, sample_every=10):
        """Full training loop"""
        print("=" * 80)
        print("Training Start")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Epochs: {num_epochs}")
        print(f"CFG probability: {self.cfg_prob}")
        
        # Early stopping info
        print(f"Early stopping patience: {self.early_stopping.patience}")
        
        # Scheduler info
        if hasattr(self.scheduler, 'base_scheduler'):
            # WarmupScheduler
            warmup_epochs = self.scheduler.warmup_epochs
            print(f"Warmup epochs: {warmup_epochs}")
            if hasattr(self.scheduler.base_scheduler, 'patience'):
                print(f"Plateau patience: {self.scheduler.base_scheduler.patience}")
        elif hasattr(self.scheduler, 'patience'):
            # ReduceLROnPlateau
            print(f"Plateau patience: {self.scheduler.patience}")
        
        print("=" * 80)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Measure epoch time
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.logger.log(train_loss, val_loss, current_lr)
            
            # Determine warmup status
            is_warmup = False
            warmup_info = ""
            if hasattr(self.scheduler, 'base_scheduler'):
                # WarmupScheduler
                is_warmup = epoch <= self.scheduler.warmup_epochs
                warmup_info = " [WARMUP]" if is_warmup else " [TRAINING]"
            
            # Scheduler step (handle different scheduler types)
            if hasattr(self.scheduler, 'step'):
                if hasattr(self.scheduler, 'base_scheduler'):
                    # WarmupScheduler with plateau
                    self.scheduler.step(val_loss)
                elif hasattr(self.scheduler, 'mode'):  # ReduceLROnPlateau
                    self.scheduler.step(val_loss)
                else:  # StepLR, CosineAnnealingLR, etc.
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}{warmup_info}")
            print(f"{'='*80}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
            
            # Scheduler patience info
            if hasattr(self.scheduler, 'base_scheduler') and hasattr(self.scheduler.base_scheduler, 'num_bad_epochs'):
                plateau_scheduler = self.scheduler.base_scheduler
                print(f"  Plateau:    {plateau_scheduler.num_bad_epochs}/{plateau_scheduler.patience} bad epochs")
            elif hasattr(self.scheduler, 'num_bad_epochs'):
                print(f"  Plateau:    {self.scheduler.num_bad_epochs}/{self.scheduler.patience} bad epochs")
            
            # Early stopping info
            print(f"  Early Stop: {self.early_stopping.counter}/{self.early_stopping.patience} patience")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"  ✓ New best model!")
            self.checkpoint.save_checkpoint(
                epoch, self.model, self.optimizer, self.scheduler,
                val_loss, self.logger.history, is_best
            )
            
            # Sample
            if epoch % sample_every == 0:
                print(f"  Generating samples...")
                self.sample_and_save(epoch)
            
            # Plot history
            self.logger.plot_history()
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n{'='*80}")
                print(f"⚠ Early stopping triggered at epoch {epoch}")
                print(f"{'='*80}")
                break
        
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)

