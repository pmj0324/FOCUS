"""
Train experiments from tasks folder.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import yaml
from pathlib import Path
import argparse
import importlib

from diffusion import GaussianDiffusion
from dataloaders import create_dataloaders
from training import DiffusionTrainer, EarlyStopping, ModelCheckpoint, Logger


def import_model_from_string(import_path):
    """Dynamically import model class from string path"""
    # Support both "SimpleUNet" and "models.unet.SimpleUNet"
    if '.' in import_path:
        # Full import path
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    else:
        # Just class name - try models package first
        try:
            from models import SimpleUNet
            model_class = SimpleUNet
        except ImportError:
            # Try to import from models module directly
            module = importlib.import_module('models')
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


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    scheduler_config = config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name', 'plateau').lower()
    
    if scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 3),
            verbose=True,
            min_lr=scheduler_config.get('min_lr', 1e-7)
        )
    elif scheduler_name == 'cosine':
        T_max = scheduler_config.get('T_max', 200)
        eta_min = scheduler_config.get('eta_min', 1e-6)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_name == 'step':
        step_size = scheduler_config.get('step_size', 50)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == 'none' or scheduler_name is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def create_dataloaders_with_split(config):
    """Create dataloaders with configurable shuffle"""
    data_dir = config['data'].get('data_dir', 'processed')
    maps_path, params_path = resolve_data_paths(data_dir)
    
    shuffle = config['data'].get('shuffle', True)
    
    return create_dataloaders(
        maps_path=maps_path,
        params_path=params_path,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        shuffle=shuffle
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
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['schedule'],
        device=device
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config['training'])
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config['training'])
    
    # Create trainer
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
    main()
