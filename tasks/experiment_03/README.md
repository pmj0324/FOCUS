# Experiment 02: Lightweight Baseline

## Overview
This experiment uses a lightweight model configuration for faster training and lower memory usage.

## Key Features

### Model Architecture
- **Base Channels**: 64 (lightweight)
- **Channel Multipliers**: [1, 2, 4, 8]
- **Time Embedding Dim**: 256
- **Total Parameters**: ~37M (lightweight)

### Training Configuration
- **Batch Size**: 32 (smaller for lighter model)
- **Learning Rate**: 1e-4 with warmup
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau
  - Patience: 2 epochs
  - Factor: 0.3 (70% LR decrease)
  - Warmup: 5 epochs (linear warmup from 0 to base LR)
- **CFG Probability**: 0.1 (10% unconditional training)
- **Gradient Clipping**: 1.0

### Enhanced Visualization
Each epoch generates:
1. **Sample Comparison** (`figs/samples_epoch_XXXX.png`)
   - 4 rows × 4 columns layout
   - Each row shows: [Real, Generated 1, Generated 2, Generated 3]
   - Real image displays cosmological parameters (Ωm, Ωb, h, ns, σ8, w0)

2. **Power Spectrum Analysis** (`figs/power_spectrum_epoch_XXXX.png`)
   - 2×2 grid showing 4 samples
   - Each plot compares real vs 3 generated samples
   - Log-log scale P(k) vs k

## Usage

### Run Training
```bash
cd /home/work/Cosmology/FOCUS/tasks/experiment_02
./run_model.sh
```

Or manually:
```bash
cd /home/work/Cosmology/FOCUS
python3 train.py --config tasks/experiment_02/config.yaml --exp_dir tasks/experiment_02
```

## Comparison with Experiment 01

| Feature | Experiment 01 (COSMO-like) | Experiment 02 (Lightweight) |
|---------|----------------------------|------------------------------|
| Base Channels | 128 | 64 |
| Parameters | ~146M | ~37M |
| Batch Size | 64 | 32 |
| Memory Usage | High | Low |
| Training Speed | Slower | Faster |
| Model Capacity | High | Medium |

## Expected Outputs

### Directory Structure
```
experiment_02/
├── config.yaml              # Configuration file
├── run_model.sh            # Training script
├── checkpoints/
│   ├── checkpoint_best.pt  # Best model
│   ├── checkpoint_last.pt  # Latest model
│   └── training_history.png  # Loss curves
├── figs/                   # Sample images
│   ├── samples_epoch_*.png # Sample comparisons
│   └── power_spectrum_epoch_*.png  # Power spectrum analysis
├── logs/
│   └── training.log        # Training log
└── README.md               # This file
```

## Notes
- **Faster Training**: ~2-3x faster than Experiment 01
- **Lower Memory**: Uses ~75% less GPU memory
- **Good for Testing**: Quick iteration and experimentation
- **Trade-off**: May have slightly lower quality than larger model