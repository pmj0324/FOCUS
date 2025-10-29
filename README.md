# FOCUS: Flow & diffusiOn model for Cosmological Universe Simulation

A modular framework for training and using diffusion models and flow matching to generate 2D dark matter maps conditioned on cosmological parameters.

## 🌟 Features

- ✅ **Modular Architecture**: Clean separation of models, training, diffusion, and utilities
- ✅ **Conditional Generation**: Generate 2D dark matter maps from 6 cosmological parameters
- ✅ **DDPM/DDIM**: Full diffusion pipeline with DDPM and fast DDIM sampling
- ✅ **Classifier-Free Guidance (CFG)**: Improve generation quality
- ✅ **Experiment Management**: Organized task structure with YAML configs
- ✅ **Easy to Extend**: Ready for flow matching, new architectures, transformers, etc.
- ✅ **Verified**: Forward diffusion converges to N(0,1) distribution

## 📁 Project Structure

```
FOCUS/
├── models/                  # Model architectures
│   ├── embeddings.py        # Time/condition embeddings
│   └── unet.py              # UNet model
│
├── dataloaders/             # Data handling
│   ├── cosmology_dataset.py # PyTorch Dataset
│   └── prepare_data.py      # Data preprocessing script
│
├── diffusion/               # Diffusion models
│   ├── schedules.py         # Noise schedules (linear, cosine, quadratic)
│   └── ddpm.py              # DDPM/DDIM implementation
│
├── flowmatching/            # Flow matching models (future)
│
├── training/                # Training utilities
│   ├── trainer.py           # Main trainer class
│   └── callbacks.py         # Early stopping, checkpointing, logging
│
├── utils/                   # Utility functions
│   ├── normalization.py     # Data normalization/denormalization
│   ├── visualization.py     # Plotting utilities
│   └── power_spectrum.py    # Power spectrum analysis
│
├── manifold_analysis/       # Manifold analysis (future)
│
├── parameter_inference/     # Parameter inference
│   ├── inference.py         # Model loading and sampling
│   └── sampling.py          # Parameter sampling utilities
│
├── tasks/                   # Experiments
│   ├── experiment_01/       # Example experiment
│   │   ├── config.yaml      # Experiment configuration
│   │   ├── checkpoints/     # Saved models
│   │   ├── logs/            # Training logs
│   │   └── figs/            # Generated figures
│   ├── train_experiment.py  # Training script
│   └── inference_experiment.py  # Inference script
│
├── configs/                 # Default configurations
│   └── default.yaml
│
├── docs/                    # Documentation
│   └── README.md            # Detailed documentation
│
├── train.py                 # Main training entry point
├── inference.py             # Main inference entry point
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies (for pip)
└── environment.yml          # Conda/micromamba environment
```

## 🚀 Quick Start

### 1. Installation

#### Option A: Using micromamba (Recommended)

Create and activate the `focus` environment using the provided `environment.yml`:

```bash
# Create environment from environment.yml
micromamba env create -f environment.yml

# Activate the environment
micromamba activate focus
```

#### Option B: Using pip

```bash
pip install -r requirements.txt
```

#### Required packages:
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `scipy >= 1.10.0`
- `tqdm >= 4.65.0`
- `pyyaml >= 6.0`

#### Replicating the environment elsewhere:

The `environment.yml` file allows you to easily recreate the environment on other systems:

```bash
# Clone the repository
git clone https://github.com/pmj0324/FOCUS.git
cd FOCUS

# Create the environment
micromamba env create -f environment.yml

# Activate the environment
micromamba activate focus
```

### 2. Prepare Data

Place your data in the following structure:
```
Data/
└── 2D/
    ├── Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy  # (15000, 256, 256)
    └── params_LH_IllustrisTNG.txt             # (1000, 6)
```

Then run:
```bash
python dataloaders/prepare_data.py --data_dir ./Data --output_dir ./processed_data
```

### 3. Run an Experiment

Create a new experiment in `tasks/`:
```bash
mkdir -p tasks/my_experiment/{checkpoints,logs,figs}
cp configs/default.yaml tasks/my_experiment/config.yaml
```

Edit `tasks/my_experiment/config.yaml` to your liking, then train:
```bash
python tasks/train_experiment.py --config tasks/my_experiment/config.yaml --exp_dir tasks/my_experiment
```

### 4. Generate Samples

After training, generate samples:
```bash
python tasks/inference_experiment.py --config tasks/my_experiment/config.yaml --exp_dir tasks/my_experiment
```

## 📋 Configuration

Edit `config.yaml` to customize:

```yaml
# Model architecture
model:
  base_channels: 64        # Model size (64/128/256)
  channel_mults: [1, 2, 4, 8]

# Diffusion schedule
diffusion:
  timesteps: 1000          # Number of diffusion steps
  schedule: "linear"       # linear/cosine/quadratic

# Training
training:
  batch_size: 2            # Adjust for GPU memory
  num_epochs: 200
  lr: 1.0e-4
  cfg_prob: 0.1           # 10% unconditional training

# Sampling
sampling:
  method: "ddim"          # ddim (fast) or ddpm (accurate)
  ddim_timesteps: 50
  cfg_scale: 2.0
```

## 🔧 Module Usage

### Models
```python
from models import SimpleUNet

model = SimpleUNet(
    in_channels=1,
    out_channels=1,
    cond_dim=6,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
)
```

### Diffusion
```python
from diffusion import GaussianDiffusion

diffusion = GaussianDiffusion(
    timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    schedule='linear',
    device='cuda'
)
```

### Training
```python
from training import DiffusionTrainer

trainer = DiffusionTrainer(
    model=model,
    diffusion=diffusion,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    lr=1e-4,
    cfg_prob=0.1,
    output_dir='./outputs'
)

trainer.train(num_epochs=200)
```

## 🎯 Future Extensions

This framework is designed to be easily extended:

1. **New Noise Schedules**: Add to `diffusion/schedules.py`
2. **New Models**: Add to `models/` (e.g., transformer-based architectures)
3. **Flow Matching**: Implement in `flowmatching/`
4. **Parameter Inference**: Add to `parameter_inference/`
5. **Manifold Analysis**: Add to `manifold_analysis/`

## 📊 Performance

| GPU | Batch Size | Training Time (100 epochs) | Memory |
|-----|-----------|----------------------------|--------|
| A100 | 16 | ~3 hours | ~14 GB |
| RTX 3090 | 8 | ~5 hours | ~8 GB |
| RTX 3080 | 4 | ~10 hours | ~6 GB |

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
training:
  batch_size: 1  # Reduce batch size
model:
  base_channels: 32  # Reduce model size
```

### Training Too Slow
```yaml
sampling:
  ddim_timesteps: 20  # Reduce sampling steps during training
```

### Poor Generation Quality
- Train longer (increase `num_epochs`)
- Increase model size (`base_channels: 128`)
- Adjust CFG scale in inference

## 📚 References

- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- **Classifier-Free Guidance**: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

## 📄 License

MIT License. See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Issues and PRs are welcome!

---

**Happy Generating!** 🚀
