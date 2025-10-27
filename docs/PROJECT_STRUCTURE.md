# FOCUS: Project Structure Guide

## 📂 Clean Organization

The project is now organized into clear, functional modules:

### Core Modules

```
focus/
│
├── 🎨 models/                # Neural network architectures
│   ├── embeddings.py         # Time & condition embeddings
│   └── unet.py              # SimpleUNet architecture
│
├── 📦 dataloaders/           # Data management
│   ├── cosmology_dataset.py # PyTorch Dataset class
│   └── prepare_data.py      # Data preprocessing
│
├── 🌊 diffusion/             # Diffusion algorithms
│   ├── schedules.py         # Noise schedules (linear/cosine/quadratic)
│   └── ddpm.py              # DDPM & DDIM implementations
│
├── 🏃 training/              # Training infrastructure
│   ├── trainer.py           # DiffusionTrainer class
│   └── callbacks.py         # EarlyStopping, Checkpointing, Logging
│
├── 🔧 utils/                 # Utility functions
│   ├── normalization.py     # Data normalization/denormalization
│   ├── visualization.py     # Plotting & visualization
│   └── power_spectrum.py    # Power spectrum analysis
│
├── 🔮 parameter_inference/   # Parameter inference tools
│   ├── inference.py         # Model loading & sampling
│   └── sampling.py          # Parameter sampling
│
├── 🌀 flowmatching/          # Flow matching (future)
│
└── 📊 manifold_analysis/     # Manifold analysis (future)
```

### Experiment Management

```
tasks/                        # All experiments here
├── experiment_01/           # Example experiment
│   ├── config.yaml          # Experiment configuration
│   ├── checkpoints/         # Model checkpoints (.pt files)
│   ├── logs/                # Training logs
│   └── figs/                # Generated figures
│
├── train_experiment.py      # Training script
└── inference_experiment.py  # Inference script
```

### Configuration

```
configs/
└── default.yaml             # Default configuration template
```

### Scripts & Testing

```
scripts/
├── visualize_data.py        # Data visualization
├── test_diffusion.py        # Test diffusion process
├── test_real_data.py        # Test with real data
├── visualize_forward_process.py  # Visualize diffusion
├── read_data.py             # Data reading utilities
├── run_all.sh               # Run all preprocessing
└── legacy/                  # Old code (for reference only)
    ├── dataset.py           # ❌ Replaced by dataloaders/
    ├── diffusion.py         # ❌ Replaced by diffusion/
    ├── model_simple.py      # ❌ Replaced by models/
    ├── utils.py             # ❌ Replaced by utils/
    └── prepare_data.py      # ❌ Replaced by dataloaders/
```

### Root Files

```
├── train.py                 # ⭐ Main training entry point
├── inference.py             # ⭐ Main inference entry point
├── setup.py                 # Package installation
├── requirements.txt         # Dependencies
├── README.md                # Main documentation
├── QUICKSTART.md            # Quick start guide
└── PROJECT_STRUCTURE.md     # This file
```

## 🚀 Quick Usage

### 1. Prepare Data
```bash
python dataloaders/prepare_data.py --data_dir ./Data --output_dir ./processed_data
```

### 2. Create New Experiment
```bash
mkdir -p tasks/my_exp/{checkpoints,logs,figs}
cp configs/default.yaml tasks/my_exp/config.yaml
# Edit tasks/my_exp/config.yaml as needed
```

### 3. Train
```bash
python train.py --config tasks/my_exp/config.yaml --exp_dir tasks/my_exp
```

### 4. Generate Samples
```bash
python inference.py --config tasks/my_exp/config.yaml --exp_dir tasks/my_exp
```

## 📝 Code Organization Principles

### ✅ DO:
- Add new models to `models/`
- Add new schedules to `diffusion/schedules.py`
- Create experiments in `tasks/`
- Use YAML configs for all experiments
- Import from packages: `from models import SimpleUNet`

### ❌ DON'T:
- Use files in `scripts/legacy/` (old versions)
- Create files in root directory
- Hard-code configurations
- Mix experiment code with core modules

## 🔄 Adding New Features

### New Model
```python
# models/my_model.py
class MyModel(nn.Module):
    ...

# models/__init__.py
from .my_model import MyModel
__all__ = [..., 'MyModel']
```

### New Noise Schedule
```python
# diffusion/schedules.py
class MySchedule(NoiseSchedule):
    def get_betas(self):
        ...
```

### New Experiment
```bash
mkdir -p tasks/new_exp/{checkpoints,logs,figs}
cp configs/default.yaml tasks/new_exp/config.yaml
# Edit config and run
python train.py --config tasks/new_exp/config.yaml --exp_dir tasks/new_exp
```

## 📦 Installation as Package

```bash
pip install -e .
```

Then use anywhere:
```python
from models import SimpleUNet
from diffusion import GaussianDiffusion
from training import DiffusionTrainer
```

## 🗑️ What Got Moved?

| Old Location | New Location | Status |
|-------------|--------------|--------|
| `dataset.py` | `dataloaders/cosmology_dataset.py` | ✅ Replaced |
| `diffusion.py` | `diffusion/ddpm.py` | ✅ Replaced |
| `model_simple.py` | `models/unet.py` | ✅ Replaced |
| `utils.py` | `utils/` package | ✅ Replaced |
| `prepare_data.py` | `dataloaders/prepare_data.py` | ✅ Replaced |
| Testing scripts | `scripts/` | ✅ Organized |

## 🎯 Next Steps

1. ✅ Clean modular structure
2. ✅ YAML-based experiment management
3. ⏳ Add flow matching to `flowmatching/`
4. ⏳ Add transformer models to `models/`
5. ⏳ Extend `parameter_inference/`
6. ⏳ Add analysis to `manifold_analysis/`

---

**Everything is now organized and ready for development!** 🚀

