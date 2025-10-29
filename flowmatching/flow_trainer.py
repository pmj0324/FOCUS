"""
Flow Matching Trainer - similar interface to DiffusionTrainer.
"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

from .flow_matching import FlowMatching


class FlowTrainer:
    """
    Trainer for Flow Matching models.
    Compatible interface with DiffusionTrainer for easy switching.
    """
    
    def __init__(
        self,
        model,
        flow_matching,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-4,
        weight_decay=1e-4,
        cfg_prob=0.1,
        output_dir='./outputs',
        optimizer=None,
        scheduler=None,
        gradient_clip=None,
    ):
        self.model = model.to(device)
        self.flow_matching = flow_matching
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg_prob = cfg_prob
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.gradient_clip = gradient_clip
        
        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
        # Scheduler
        self.scheduler = scheduler
        
        # Early stopping (import from training module)
        try:
            from training.callbacks import EarlyStopping
            self.early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        except:
            # Simple early stopping if import fails
            class SimpleEarlyStopping:
                def __init__(self, patience=10):
                    self.patience = patience
                    self.counter = 0
                    self.best_loss = float('inf')
                def __call__(self, val_loss):
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.counter = 0
                        return False
                    else:
                        self.counter += 1
                        return self.counter >= self.patience
            self.early_stopping = SimpleEarlyStopping(patience=10)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, params in pbar:
            images = images.to(self.device)
            params = params.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Flow Matching loss
            loss = self.flow_matching.compute_flow_loss(
                self.model, images, params, self.cfg_prob
            )
            
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for images, params in self.val_loader:
            images = images.to(self.device)
            params = params.to(self.device)
            
            loss = self.flow_matching.compute_flow_loss(
                self.model, images, params, cfg_prob=0.0  # No CFG during validation
            )
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    @torch.no_grad()
    def sample_and_save(self, epoch, num_samples=4, cfg_scale=2.0):
        """Generate and save samples with real comparison"""
        self.model.eval()
        
        # Get real samples from validation set
        real_images, real_params = next(iter(self.val_loader))
        real_images = real_images[:num_samples].to(self.device)
        real_params = real_params[:num_samples].to(self.device)
        
        # Check if power spectrum utilities are available
        try:
            from utils.power_spectrum import compute_power_spectrum
            has_power_spectrum = True
        except ImportError:
            has_power_spectrum = False
        
        # Generate multiple samples per condition
        num_gen_per_cond = 3
        all_generated = []
        
        for _ in range(num_gen_per_cond):
            generated = self.flow_matching.sample(
                self.model,
                shape=(num_samples, 1, 256, 256),
                cond=real_params,
                cfg_scale=cfg_scale,
                progress=False
            )
            all_generated.append(generated)
        
        # Save sample comparison
        fig, axes = plt.subplots(num_samples, num_gen_per_cond + 1, 
                                figsize=((num_gen_per_cond + 1) * 3, num_samples * 3))
        
        for i in range(num_samples):
            # Real image
            axes[i, 0].imshow(real_images[i, 0].cpu().numpy(), cmap='viridis', origin='lower')
            axes[i, 0].set_title('Real', fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Cosmological parameters
            param_text = ', '.join([f'{p:.3f}' for p in real_params[i].cpu().numpy()])
            axes[i, 0].text(0.5, -0.05, param_text, transform=axes[i, 0].transAxes,
                          ha='center', va='top', fontsize=8, color='blue')
            
            # Generated images
            for j in range(num_gen_per_cond):
                axes[i, j+1].imshow(all_generated[j][i, 0].cpu().numpy(), 
                                   cmap='viridis', origin='lower')
                axes[i, j+1].set_title(f'Flow Gen {j+1}', fontsize=10)
                axes[i, j+1].axis('off')
        
        plt.suptitle(f'Flow Matching - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir.parent / 'figs' / f'samples_epoch_{epoch:04d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Power spectrum comparison
        if has_power_spectrum:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(min(4, num_samples)):
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
                             label=f'Flow Gen {j+1}', alpha=0.7)
                
                ax.set_xlabel('k [h Mpc$^{-1}$]', fontsize=10)
                ax.set_ylabel('P(k)', fontsize=10)
                ax.set_title(f'Sample {i+1} Power Spectrum', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir.parent / 'figs' / f'power_spectrum_epoch_{epoch:04d}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'history': self.history,
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.output_dir / 'checkpoint_last.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(self.history['lr'], linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, num_epochs=100, sample_every=10):
        """Full training loop"""
        print("=" * 80)
        print("Flow Matching Training Start")
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
        if self.scheduler is not None:
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
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            # Determine warmup status
            is_warmup = False
            warmup_info = ""
            if self.scheduler is not None and hasattr(self.scheduler, 'base_scheduler'):
                # WarmupScheduler
                is_warmup = epoch <= self.scheduler.warmup_epochs
                warmup_info = " [WARMUP]" if is_warmup else " [TRAINING]"
            
            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    # Check if it's a custom WarmupScheduler
                    if hasattr(self.scheduler, 'base_scheduler'):
                        self.scheduler.step(val_loss)
                    elif hasattr(self.scheduler, 'mode'):  # ReduceLROnPlateau
                        self.scheduler.step(val_loss)
                    else:  # Other schedulers
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
            if self.scheduler is not None:
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
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Sample
            if epoch % sample_every == 0:
                print(f"  Generating samples...")
                self.sample_and_save(epoch, num_samples=4, cfg_scale=2.0)
            
            # Plot history
            if epoch % 10 == 0:
                self.plot_history()
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n{'='*80}")
                print(f"⚠ Early stopping triggered at epoch {epoch}")
                print(f"{'='*80}")
                break
        
        print("\n" + "=" * 80)
        print("✅ Flow Matching Training Complete!")
        print("=" * 80)
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)

