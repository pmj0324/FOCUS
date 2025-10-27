"""
Training callbacks.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class ModelCheckpoint:
    """Model checkpoint saver"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.best_val_loss = float('inf')
        
    def save_checkpoint(self, epoch, model, optimizer, scheduler, val_loss, history, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'history': history,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'checkpoint_last.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pt')
            self.best_val_loss = val_loss
            print(f"✓ Best model saved! (Val Loss: {val_loss:.4f})")


class Logger:
    """Training logger"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
    def log(self, train_loss, val_loss, lr):
        """Log metrics"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(lr)
        
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1].plot(self.history['lr'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()

