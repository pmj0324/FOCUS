"""
Main training script with integrated experiment functionality.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import yaml
from pathlib import Path
import argparse
import importlib
import sys

from diffusion import GaussianDiffusion
from dataloaders import create_dataloaders
from training import DiffusionTrainer, EarlyStopping, ModelCheckpoint, Logger
from flowmatching import FlowMatching, FlowTrainer, FlowUNet


def import_model_from_string(import_path):
    """Dynamically import model class from string path"""
    # Support both "SimpleUNet", "FlowUNet" and "models.unet.SimpleUNet"
    if '.' in import_path:
        # Full import path
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    else:
        # Just class name - check for flow or diffusion models
        if import_path == "FlowUNet":
            from flowmatching import FlowUNet
            model_class = FlowUNet
        elif import_path == "SimpleUNet":
            from models import SimpleUNet
            model_class = SimpleUNet
        else:
            # Try to import from models module directly
            try:
                module = importlib.import_module('models')
                model_class = getattr(module, import_path)
            except:
                # Try flowmatching module
                module = importlib.import_module('flowmatching')
                model_class = getattr(module, import_path)
    
    return model_class


def resolve_data_paths(data_dir):
    """Resolve data paths, support 'processed' shortcut"""
    if data_dir == "processed":
        base_dir = Path("./processed_data")
        maps_path = str(base_dir / "maps_normalized.npy")
        params_path = str(base_dir / "params_normalized.npy")
    else:
        # Use provided paths directly
        data_config = data_dir if isinstance(data_dir, dict) else {}
        maps_path = data_config.get('maps_path', f"{data_dir}/maps_normalized.npy")
        params_path = data_config.get('params_path', f"{data_dir}/params_normalized.npy")
    
    return maps_path, params_path


def create_optimizer(model, config):
    """Create optimizer based on config"""
    opt_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif opt_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer


class WarmupScheduler:
    """Learning rate warmup wrapper"""
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, val_loss=None):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.base_scheduler is not None:
            # Use base scheduler after warmup
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if val_loss is not None:
                    self.base_scheduler.step(val_loss)
                    # Apply the learning rate set by the plateau scheduler
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.base_scheduler._last_lr[0]
            else:
                self.base_scheduler.step()
    
    def state_dict(self):
        state = {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
        }
        if self.base_scheduler is not None:
            state['base_scheduler'] = self.base_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        if self.base_scheduler is not None and 'base_scheduler' in state_dict:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    scheduler_config = config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name', 'plateau').lower()
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    
    # Create base scheduler
    if scheduler_name == 'plateau':
        base_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.3),  # 70% decrease = 0.3 factor
            patience=scheduler_config.get('patience', 2),
            verbose=True,
            min_lr=scheduler_config.get('min_lr', 1e-7)
        )
    elif scheduler_name == 'cosine':
        T_max = scheduler_config.get('T_max', 200)
        eta_min = scheduler_config.get('eta_min', 1e-6)
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_name == 'step':
        step_size = scheduler_config.get('step_size', 50)
        gamma = scheduler_config.get('gamma', 0.1)
        base_scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == 'none' or scheduler_name is None:
        base_scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    # Wrap with warmup if needed
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
    else:
        scheduler = base_scheduler
    
    return scheduler


def create_dataloaders_with_split(config):
    """Create dataloaders with configurable shuffle"""
    # Use direct paths if provided, otherwise fall back to data_dir shortcut
    if 'maps_path' in config['data'] and 'params_path' in config['data']:
        maps_path = config['data']['maps_path']
        params_path = config['data']['params_path']
    else:
        data_dir = config['data'].get('data_dir', 'processed')
        maps_path, params_path = resolve_data_paths(data_dir)
    
    shuffle = config['data'].get('shuffle', True)

    # Augmentation settings (optional)
    training_cfg = config.get('training', {})
    use_augmentation = training_cfg.get('use_augmentation', False)
    augmentation_config = training_cfg.get('augmentation', None)

    return create_dataloaders(
        maps_path=maps_path,
        params_path=params_path,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        shuffle=shuffle,
        use_augmentation=use_augmentation,
        augmentation_config=augmentation_config
    )


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'figs').mkdir(exist_ok=True)
    
    # Device
    device = config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders_with_split(config)
    
    # Create model dynamically
    model_class = import_model_from_string(config['model']['from'])
    model_args = {k: v for k, v in config['model'].items() if k != 'from'}
    model = model_class(**model_args)
    
    # Create optimizer
    optimizer = create_optimizer(model, config['training'])
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config['training'])
    
    # Determine training method from config
    method = config.get('method', 'diffusion').lower()
    
    if method == 'flow':
        # Flow Matching training
        print("=" * 80)
        print("Using Flow Matching")
        print("=" * 80)
        
        # Create Flow Matching
        flow_config = config.get('flow_matching', {})
        flow_matching = FlowMatching(
            sigma_min=flow_config.get('sigma_min', 0.0),
            sigma_max=flow_config.get('sigma_max', 1.0),
            device=device
        )
        
        # Create Flow Trainer
        trainer = FlowTrainer(
            model=model,
            flow_matching=flow_matching,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'],
            cfg_prob=config['training']['cfg_prob'],
            output_dir=exp_dir / 'checkpoints',
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip=config['training'].get('gradient_clip', None),
        )
        
    else:
        # Diffusion training (default)
        print("=" * 80)
        print("Using Diffusion")
        print("=" * 80)
        
        # Create diffusion
        diffusion = GaussianDiffusion(
            timesteps=config['diffusion']['timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            schedule=config['diffusion']['schedule'],
            device=device
        )
        
        # Create Diffusion Trainer
        trainer = DiffusionTrainer(
            model=model,
            diffusion=diffusion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'],
            cfg_prob=config['training']['cfg_prob'],
            output_dir=exp_dir / 'checkpoints',
            optimizer=optimizer,
            scheduler=scheduler,
        )
    
    # Train
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        sample_every=config['training']['sample_every']
    )


if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Usage: python train.py --config <config.yaml> --exp_dir <exp_dir>")
        print("\nExample:")
        print("  python train.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01")
        sys.exit(1)
    
    main()
