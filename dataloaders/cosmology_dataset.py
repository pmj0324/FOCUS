"""
Dataset for cosmology dark matter maps.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CosmologyDataset(Dataset):
    """
    Dark matter maps + cosmological parameters dataset
    
    Supports data augmentation for training.
    """
    def __init__(self, maps_path, params_path, transform=None, augmentation=None):
        """
        Args:
            maps_path: Path to normalized maps .npy file
            params_path: Path to normalized params .npy file
            transform: Optional additional transform (deprecated, use augmentation)
            augmentation: Data augmentation transform (e.g., CosmologyAugmentation)
        """
        self.maps = np.load(maps_path)  # (N, 256, 256)
        self.params = np.load(params_path)  # (N, 6)
        self.transform = transform  # For backward compatibility
        self.augmentation = augmentation
        
        assert len(self.maps) == len(self.params), "Maps and params counts don't match!"
        
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        # Map: (256, 256) -> (1, 256, 256) add channel dimension
        map_data = torch.FloatTensor(self.maps[idx]).unsqueeze(0)
        
        # Params: (6,)
        params = torch.FloatTensor(self.params[idx])
        
        # Apply augmentation (for training)
        if self.augmentation:
            map_data = self.augmentation(map_data)
        
        # Apply transform (backward compatibility)
        if self.transform:
            map_data = self.transform(map_data)
            
        return map_data, params


def create_dataloaders(
    maps_path, 
    params_path, 
    batch_size=16, 
    train_split=0.9, 
    num_workers=4, 
    shuffle=True,
    use_augmentation=False,
    augmentation_config=None
):
    """
    Create train/validation dataloaders with optional data augmentation.
    
    Args:
        maps_path: Maps file path
        params_path: Params file path
        batch_size: Batch size
        train_split: Training data ratio
        num_workers: Number of data loading workers
        shuffle: Whether to use random shuffle for train/val split (default: True)
        use_augmentation: Whether to use data augmentation for training (default: False)
        augmentation_config: Dict with augmentation config (rotation_p, flip_p, etc.)
        
    Returns:
        train_loader, val_loader
    """
    from .augmentation import get_train_augmentation, get_val_augmentation
    
    # Prepare augmentation
    if use_augmentation:
        if augmentation_config is None:
            augmentation_config = {'rotation_p': 0.75, 'flip_p': 0.5}
        
        train_aug = get_train_augmentation(**augmentation_config)
        val_aug = get_val_augmentation()
        
        print(f"✓ Data augmentation enabled:")
        print(f"  {train_aug}")
    else:
        train_aug = None
        val_aug = None
        print(f"✗ Data augmentation disabled")
    
    # Create separate datasets for train and val with different augmentations
    train_dataset_full = CosmologyDataset(maps_path, params_path, augmentation=train_aug)
    val_dataset_full = CosmologyDataset(maps_path, params_path, augmentation=val_aug)
    
    # Train/Val split
    total_size = len(train_dataset_full)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    if shuffle:
        # Get indices
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
    else:
        # Sequential split
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
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

