"""
Dataset for cosmology dark matter maps.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CosmologyDataset(Dataset):
    """
    Dark matter maps + cosmological parameters dataset
    """
    def __init__(self, maps_path, params_path, transform=None):
        """
        Args:
            maps_path: Path to normalized maps .npy file
            params_path: Path to normalized params .npy file
            transform: Optional additional transform
        """
        self.maps = np.load(maps_path)  # (N, 256, 256)
        self.params = np.load(params_path)  # (N, 6)
        self.transform = transform
        
        assert len(self.maps) == len(self.params), "Maps and params counts don't match!"
        
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        # Map: (256, 256) -> (1, 256, 256) add channel dimension
        map_data = torch.FloatTensor(self.maps[idx]).unsqueeze(0)
        
        # Params: (6,)
        params = torch.FloatTensor(self.params[idx])
        
        if self.transform:
            map_data = self.transform(map_data)
            
        return map_data, params


def create_dataloaders(maps_path, params_path, batch_size=16, train_split=0.9, num_workers=4, shuffle=True):
    """
    Create train/validation dataloaders
    
    Args:
        maps_path: Maps file path
        params_path: Params file path
        batch_size: Batch size
        train_split: Training data ratio
        num_workers: Number of data loading workers
        shuffle: Whether to use random shuffle for train/val split (default: True)
        
    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = CosmologyDataset(maps_path, params_path)
    
    # Train/Val split
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    if shuffle:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducibility
        )
    else:
        # Sequential split without shuffling
        indices = list(range(total_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split complete:")
    print(f"  - Train: {train_size} samples")
    print(f"  - Val: {val_size} samples")
    
    return train_loader, val_loader

