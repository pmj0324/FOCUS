"""
Data Augmentation for Cosmological 2D Maps

우주론적 2D dark matter maps를 위한 data augmentation.
우주의 등방성(isotropy) 때문에 rotation과 flip은 물리적으로 타당한 augmentation입니다.

Supported Augmentations:
1. Random rotation (90, 180, 270 degrees)
2. Random horizontal flip
3. Random vertical flip
4. Combination of above

Author: Data Augmentation Module
Date: 2025-10-29
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple


class RandomRotation90:
    """
    Randomly rotate the image by k*90 degrees (k=0,1,2,3).
    
    이 augmentation은 우주론적으로 타당합니다:
    - 우주는 등방성(isotropic)이므로 방향에 무관
    - 90도 배수 회전은 픽셀 정보 손실 없음
    """
    
    def __init__(self, p: float = 0.75):
        """
        Args:
            p: Probability of applying rotation (default: 0.75)
        """
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random rotation.
        
        Args:
            x: Input tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Rotated tensor with same shape
        """
        if torch.rand(1).item() > self.p:
            return x
        
        # Random rotation strictly among 90°, 180°, 270° (exclude 0°)
        # Map indices 0,1,2 -> k=1,2,3
        k = torch.randint(1, 4, (1,)).item()
        
        # torch.rot90: rotate in the plane of last two dimensions
        # dims=[H, W] for shape [C, H, W] or [B, C, H, W]
        dims = (-2, -1)  # (H, W)
        return torch.rot90(x, k=k, dims=dims)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomFlip:
    """
    Randomly flip the image horizontally and/or vertically.
    
    우주론적으로 타당한 이유:
    - 우주 구조는 좌우/상하 대칭에 대해 불변
    - Parity symmetry
    """
    
    def __init__(
        self, 
        horizontal_p: float = 0.5,
        vertical_p: float = 0.5
    ):
        """
        Args:
            horizontal_p: Probability of horizontal flip
            vertical_p: Probability of vertical flip
        """
        self.horizontal_p = horizontal_p
        self.vertical_p = vertical_p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random flips.
        
        Args:
            x: Input tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Flipped tensor with same shape
        """
        # Horizontal flip (along width)
        if torch.rand(1).item() < self.horizontal_p:
            x = torch.flip(x, dims=[-1])  # flip last dimension (W)
        
        # Vertical flip (along height)
        if torch.rand(1).item() < self.vertical_p:
            x = torch.flip(x, dims=[-2])  # flip second-to-last dimension (H)
        
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}(h_p={self.horizontal_p}, v_p={self.vertical_p})"


class Compose:
    """
    Compose multiple augmentations together.
    """
    
    def __init__(self, transforms: List):
        """
        Args:
            transforms: List of augmentation transforms
        """
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms sequentially.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class CosmologyAugmentation:
    """
    Complete augmentation pipeline for cosmological maps.
    
    단계적 적용:
    1. Random rotation (90°, 180°, 270°)
    2. Random horizontal flip
    3. Random vertical flip
    
    각 단계는 독립적으로 적용됩니다.
    """
    
    def __init__(
        self,
        rotation_p: float = 0.75,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        use_rotation: bool = True,
        use_flip: bool = True
    ):
        """
        Args:
            rotation_p: Probability of rotation (90°, 180°, 270° 중 랜덤)
            horizontal_flip_p: Probability of horizontal flip
            vertical_flip_p: Probability of vertical flip
            use_rotation: Whether to use rotation
            use_flip: Whether to use flip
        """
        self.rotation_p = rotation_p
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.use_rotation = use_rotation
        self.use_flip = use_flip
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline step by step.
        
        Args:
            x: Input tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            Augmented tensor
        """
        # Step 1: Random rotation (90°, 180°, 270°)
        if self.use_rotation and torch.rand(1).item() < self.rotation_p:
            # Random rotation strictly among 90°, 180°, 270°
            k = torch.randint(1, 4, (1,)).item()  # k=1,2,3
            dims = (-2, -1)  # (H, W)
            x = torch.rot90(x, k=k, dims=dims)
        
        # Step 2: Random horizontal flip
        if self.use_flip and torch.rand(1).item() < self.horizontal_flip_p:
            x = torch.flip(x, dims=[-1])  # flip last dimension (W)
        
        # Step 3: Random vertical flip
        if self.use_flip and torch.rand(1).item() < self.vertical_flip_p:
            x = torch.flip(x, dims=[-2])  # flip second-to-last dimension (H)
        
        return x
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"rotation_p={self.rotation_p}, "
                f"h_flip_p={self.horizontal_flip_p}, "
                f"v_flip_p={self.vertical_flip_p})")


class NoAugmentation:
    """
    Dummy augmentation that does nothing.
    Useful for validation/test sets.
    """
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


def get_train_augmentation(
    rotation_p: float = 0.75,
    flip_p: float = 0.5,
    **kwargs
) -> CosmologyAugmentation:
    """
    Get default training augmentation.
    
    Args:
        rotation_p: Rotation probability
        flip_p: Flip probability (both H and V)
        
    Returns:
        CosmologyAugmentation instance
    """
    return CosmologyAugmentation(
        rotation_p=rotation_p,
        horizontal_flip_p=flip_p,
        vertical_flip_p=flip_p,
        use_rotation=True,
        use_flip=True
    )


def get_val_augmentation() -> NoAugmentation:
    """
    Get validation augmentation (no augmentation).
    
    Returns:
        NoAugmentation instance
    """
    return NoAugmentation()


# Visualization helper
def visualize_augmentations(
    image: torch.Tensor,
    augmentation: CosmologyAugmentation,
    num_samples: int = 8
) -> List[torch.Tensor]:
    """
    Generate multiple augmented versions of an image for visualization.
    
    Args:
        image: Input image [C, H, W]
        augmentation: Augmentation transform
        num_samples: Number of augmented samples to generate
        
    Returns:
        List of augmented images
    """
    augmented_images = []
    for _ in range(num_samples):
        aug_img = augmentation(image.clone())
        augmented_images.append(aug_img)
    return augmented_images


if __name__ == '__main__':
    """
    Test augmentation functions.
    """
    print("Testing Cosmology Augmentation...")
    print("="*60)
    
    # Create dummy data
    dummy_image = torch.randn(1, 256, 256)
    print(f"Input shape: {dummy_image.shape}")
    
    # Test individual transforms
    print("\n1. RandomRotation90:")
    rot = RandomRotation90(p=1.0)
    for i in range(4):
        rotated = rot(dummy_image)
        print(f"   Sample {i+1}: shape={rotated.shape}")
    
    print("\n2. RandomFlip:")
    flip = RandomFlip(horizontal_p=0.5, vertical_p=0.5)
    for i in range(4):
        flipped = flip(dummy_image)
        print(f"   Sample {i+1}: shape={flipped.shape}")
    
    print("\n3. CosmologyAugmentation (combined):")
    aug = CosmologyAugmentation(
        rotation_p=0.75,
        horizontal_flip_p=0.5,
        vertical_flip_p=0.5
    )
    print(f"   {aug}")
    
    for i in range(4):
        augmented = aug(dummy_image)
        print(f"   Sample {i+1}: shape={augmented.shape}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")


