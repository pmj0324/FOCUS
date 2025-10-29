# Experiment 01: COSMO Baseline

## Overview
This experiment replicates the COSMO model configuration with enhanced visualization and monitoring.

## Key Features

### Model Architecture
- **Base Channels**: 128 (same as COSMO)
- **Channel Multipliers**: [1, 2, 4, 8]
- **Time Embedding Dim**: 256
- **Total Parameters**: ~50M (comparable to COSMO)

### Training Configuration
- **Batch Size**: 64
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
1. **Sample Comparison** (`samples_epoch_XXXX.png`)
   - 4 rows × 4 columns layout
   - Each row shows: [Real, Generated 1, Generated 2, Generated 3]
   - Real image displays cosmological parameters (Ωm, Ωb, h, ns, σ8, w0)

2. **Power Spectrum Analysis** (`power_spectrum_epoch_XXXX.png`)
   - 2×2 grid showing 4 samples
   - Each plot compares real vs 3 generated samples
   - Log-log scale P(k) vs k

### Differences from Original COSMO
1. **Integrated Architecture**: Single codebase instead of separate train_experiment.py
2. **Enhanced Monitoring**: 
   - Real vs generated comparison every epoch
   - Power spectrum analysis
   - Parameter display on images
3. **Flexible Configuration**: YAML-based config with easy modification
4. **Warmup Scheduler**: Better convergence at training start

## Usage

### Run Training
```bash
cd /home/work/Cosmology/FOCUS/tasks/experiment_01
./run_model.sh
```

Or manually:
```bash
cd /home/work/Cosmology/FOCUS
python3 train.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01
```

### Monitor Training
```bash
# View logs
tail -f /home/work/Cosmology/FOCUS/tasks/experiment_01/logs/training.log

# Check latest samples
ls -lt /home/work/Cosmology/FOCUS/tasks/experiment_01/checkpoints/samples_*.png | head
```

## Expected Outputs

### Directory Structure
```
experiment_01/
├── config.yaml              # Configuration file
├── run_model.sh            # Training script
├── checkpoints/
│   ├── checkpoint_best.pt  # Best model
│   ├── checkpoint_last.pt  # Latest model
│   ├── samples_epoch_*.png # Sample comparisons
│   ├── power_spectrum_epoch_*.png  # Power spectrum analysis
│   └── training_history.png  # Loss curves
├── logs/
│   └── training.log        # Training log
└── figs/                   # Additional figures
```

### Checkpoints
Checkpoints contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Scheduler state
- `epoch`: Current epoch
- `val_loss`: Validation loss
- `history`: Training history (train_loss, val_loss, lr)

## Comparison with COSMO

| Feature | COSMO | FOCUS Experiment 01 |
|---------|-------|---------------------|
| Base Channels | 128 | 128 |
| Batch Size | 64 | 64 |
| Learning Rate | 1e-4 | 1e-4 |
| Scheduler | Plateau (patience=5, factor=0.5) | Plateau (patience=2, factor=0.3) |
| Warmup | None | 5 epochs |
| Sampling Frequency | Every 2 epochs | Every epoch |
| Visualization | Simple 4-panel | Real + 3 Generated + Parameters + Power Spectrum |

## Notes
- Training takes ~8-10 hours on a single GPU for 200 epochs
- Sample every epoch with detailed comparison for better monitoring
- Power spectrum analysis helps validate physical correctness
- Warmup prevents initial instability

