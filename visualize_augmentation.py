"""
Visualize Data Augmentation Effects

이 스크립트는 data augmentation이 cosmology maps에 어떻게 적용되는지 시각화합니다.

Usage:
    python visualize_augmentation.py
    python visualize_augmentation.py --sample_idx 100 --num_augs 8
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dataloaders.cosmology_dataset import CosmologyDataset
from dataloaders.augmentation import (
    CosmologyAugmentation,
    RandomRotation90,
    RandomFlip,
    visualize_augmentations
)


def plot_augmentation_comparison(
    original_image: torch.Tensor,
    augmented_images: list,
    save_path: str = None,
    title: str = "Data Augmentation Examples"
):
    """
    원본 이미지와 augmented 이미지들을 비교 시각화.
    
    Args:
        original_image: 원본 이미지 [1, H, W]
        augmented_images: Augmented 이미지 리스트
        save_path: 저장 경로
        title: 제목
    """
    num_aug = len(augmented_images)
    n_cols = 4
    n_rows = (num_aug + 1 + n_cols - 1) // n_cols  # +1 for original
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot original
    ax = axes[0]
    im = ax.imshow(original_image[0].cpu().numpy(), cmap='viridis')
    ax.set_title('Original', fontsize=14, fontweight='bold', color='red')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot augmented versions
    for i, aug_img in enumerate(augmented_images):
        ax = axes[i + 1]
        im = ax.imshow(aug_img[0].cpu().numpy(), cmap='viridis')
        ax.set_title(f'Augmented {i+1}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Remove extra subplots
    for i in range(num_aug + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
    return fig


def plot_rotation_examples(
    original_image: torch.Tensor,
    save_path: str = None
):
    """
    90도 회전의 모든 경우를 시각화.
    
    Args:
        original_image: 원본 이미지 [1, H, W]
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°
    titles = ['0° (Original)', '90° CW', '180°', '270° CW']
    
    for i, (k, title) in enumerate(zip(rotations, titles)):
        if k == 0:
            img = original_image
        else:
            img = torch.rot90(original_image, k=k, dims=(-2, -1))
        
        ax = axes[i]
        im = ax.imshow(img[0].cpu().numpy(), cmap='viridis')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Random Rotation (k×90°)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
    return fig


def plot_flip_examples(
    original_image: torch.Tensor,
    save_path: str = None
):
    """
    Flip의 모든 경우를 시각화.
    
    Args:
        original_image: 원본 이미지 [1, H, W]
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original
    ax = axes[0, 0]
    im = ax.imshow(original_image[0].cpu().numpy(), cmap='viridis')
    ax.set_title('Original', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Horizontal flip
    ax = axes[0, 1]
    img_h = torch.flip(original_image, dims=[-1])
    im = ax.imshow(img_h[0].cpu().numpy(), cmap='viridis')
    ax.set_title('Horizontal Flip', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Vertical flip
    ax = axes[1, 0]
    img_v = torch.flip(original_image, dims=[-2])
    im = ax.imshow(img_v[0].cpu().numpy(), cmap='viridis')
    ax.set_title('Vertical Flip', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Both
    ax = axes[1, 1]
    img_hv = torch.flip(torch.flip(original_image, dims=[-1]), dims=[-2])
    im = ax.imshow(img_hv[0].cpu().numpy(), cmap='viridis')
    ax.set_title('Horizontal + Vertical Flip', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Random Flip Variations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Data Augmentation')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                       help='Data directory')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--num_augs', type=int, default=7,
                       help='Number of augmented samples to generate')
    parser.add_argument('--rotation_p', type=float, default=0.75,
                       help='Rotation probability')
    parser.add_argument('--flip_p', type=float, default=0.5,
                       help='Flip probability')
    parser.add_argument('--output_dir', type=str, default='./augmentation_vis',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Data Augmentation Visualization")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    data_dir = Path(args.data_dir)
    maps_path = data_dir / 'maps_normalized.npy'
    params_path = data_dir / 'params_normalized.npy'
    
    if not maps_path.exists():
        print(f"Error: {maps_path} not found!")
        return
    
    print(f"\nLoading data from {data_dir}")
    dataset = CosmologyDataset(str(maps_path), str(params_path))
    
    # Get sample
    sample_idx = args.sample_idx
    if sample_idx >= len(dataset):
        print(f"Warning: sample_idx {sample_idx} out of range, using 0")
        sample_idx = 0
    
    original_image, params = dataset[sample_idx]
    print(f"\nSample {sample_idx}:")
    print(f"  Image shape: {original_image.shape}")
    print(f"  Parameters: {params.numpy()}")
    
    # Create augmentation
    augmentation = CosmologyAugmentation(
        rotation_p=args.rotation_p,
        horizontal_flip_p=args.flip_p,
        vertical_flip_p=args.flip_p,
        use_rotation=True,
        use_flip=True
    )
    
    print(f"\nAugmentation configuration:")
    print(f"  {augmentation}")
    
    # Generate augmented samples
    print(f"\nGenerating {args.num_augs} augmented samples...")
    augmented_images = visualize_augmentations(
        original_image,
        augmentation,
        num_samples=args.num_augs
    )
    
    # Plot 1: Combined comparison
    print("\n1. Plotting combined comparison...")
    plot_augmentation_comparison(
        original_image,
        augmented_images,
        save_path=output_dir / f'augmentation_comparison_idx{sample_idx}.png',
        title=f'Data Augmentation Examples (Sample {sample_idx})'
    )
    
    # Plot 2: Rotation examples
    print("\n2. Plotting rotation examples...")
    plot_rotation_examples(
        original_image,
        save_path=output_dir / f'rotation_examples_idx{sample_idx}.png'
    )
    
    # Plot 3: Flip examples
    print("\n3. Plotting flip examples...")
    plot_flip_examples(
        original_image,
        save_path=output_dir / f'flip_examples_idx{sample_idx}.png'
    )
    
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - augmentation_comparison_idx{sample_idx}.png")
    print(f"  - rotation_examples_idx{sample_idx}.png")
    print(f"  - flip_examples_idx{sample_idx}.png")


if __name__ == '__main__':
    main()




