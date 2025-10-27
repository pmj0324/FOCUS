"""
Data preparation: Repeat params 15 times to match maps.
"""
import numpy as np
import os
from pathlib import Path


def prepare_cosmology_data(data_dir, output_dir):
    """
    Prepare cosmology data for training.
    
    Args:
        data_dir: Directory containing original data
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Loading data...")
    print("=" * 80)
    
    # Load original data
    params_path = os.path.join(data_dir, "2D", "params_LH_IllustrisTNG.txt")
    maps_path = os.path.join(data_dir, "2D", "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy")
    
    params = np.loadtxt(params_path)
    maps = np.load(maps_path)
    
    print(f"Original params shape: {params.shape}")  # (1000, 6)
    print(f"Original maps shape: {maps.shape}")      # (15000, 256, 256)
    
    # Repeat params 15 times
    params_expanded = np.repeat(params, 15, axis=0)
    print(f"\nExpanded params shape: {params_expanded.shape}")  # (15000, 6)
    
    # Check matching
    assert params_expanded.shape[0] == maps.shape[0], "Sample counts don't match!"
    print(f"✓ Data matching complete: {params_expanded.shape[0]} samples")
    
    print("\n" + "=" * 80)
    print("Normalizing data...")
    print("=" * 80)
    
    # Maps log scale normalization (handle large value range)
    print(f"Maps original - Min: {maps.min():.2e}, Max: {maps.max():.2e}")
    maps_log = np.log10(maps + 1e-10)  # log scale
    print(f"Maps log - Min: {maps_log.min():.4f}, Max: {maps_log.max():.4f}")
    
    # Normalize to [-1, 1] range
    maps_min, maps_max = maps_log.min(), maps_log.max()
    maps_normalized = 2 * (maps_log - maps_min) / (maps_max - maps_min) - 1
    print(f"Maps normalized - Min: {maps_normalized.min():.4f}, Max: {maps_normalized.max():.4f}")
    
    # Params normalization ([0, 1] range)
    params_min = params_expanded.min(axis=0)
    params_max = params_expanded.max(axis=0)
    params_normalized = (params_expanded - params_min) / (params_max - params_min)
    print(f"\nParams normalization complete")
    print(f"Parameter ranges: {params_normalized.min(axis=0)} ~ {params_normalized.max(axis=0)}")
    
    print("\n" + "=" * 80)
    print("Saving data...")
    print("=" * 80)
    
    # Save
    np.save(os.path.join(output_dir, "maps_normalized.npy"), maps_normalized.astype(np.float32))
    np.save(os.path.join(output_dir, "params_normalized.npy"), params_normalized.astype(np.float32))
    
    # Save normalization statistics (for denormalization)
    normalization_stats = {
        'maps_min': maps_min,
        'maps_max': maps_max,
        'params_min': params_min,
        'params_max': params_max
    }
    np.save(os.path.join(output_dir, "normalization_stats.npy"), normalization_stats)
    
    print(f"✓ Save complete:")
    print(f"  - maps_normalized.npy: {maps_normalized.shape}")
    print(f"  - params_normalized.npy: {params_normalized.shape}")
    print(f"  - normalization_stats.npy")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Original data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    args = parser.parse_args()
    
    prepare_cosmology_data(args.data_dir, args.output_dir)

