"""
Diffusion model trainer.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
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
        """Generate and save samples"""
        self.model.eval()
        
        # Get conditions from validation data
        _, params = next(iter(self.val_loader))
        params = params[:num_samples].to(self.device)
        
        # DDIM sampling
        samples = self.diffusion.ddim_sample(
            self.model,
            shape=(num_samples, 1, 256, 256),
            cond=params,
            ddim_timesteps=50,
            cfg_scale=cfg_scale,
            progress=False
        )
        
        # Visualize
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
        
        for i in range(num_samples):
            sample = samples[i, 0].cpu().numpy()
            axes[i].imshow(sample, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'samples_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
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
        print("=" * 80)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.logger.log(train_loss, val_loss, current_lr)
            
            # Scheduler step (handle different scheduler types)
            if hasattr(self.scheduler, 'step'):
                if hasattr(self.scheduler, 'mode'):  # ReduceLROnPlateau
                    self.scheduler.step(val_loss)
                else:  # StepLR, CosineAnnealingLR, etc.
                    self.scheduler.step()
            
            # Print
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
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
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)

