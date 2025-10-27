# FOCUS: Quick Start Guide

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python dataloaders/prepare_data.py --data_dir ./Data --output_dir ./processed_data
```

### 3. Run Training
```bash
python train.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01
```

### 4. Generate Samples
```bash
python inference.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01
```

## 📁 Creating Your Own Experiment

### Step 1: Create Experiment Directory
```bash
mkdir -p tasks/my_experiment/{checkpoints,logs,figs}
cp configs/default.yaml tasks/my_experiment/config.yaml
```

### Step 2: Edit Configuration
Edit `tasks/my_experiment/config.yaml`:

```yaml
# Example: Larger model
model:
  base_channels: 128  # Increase model size

# Example: Cosine schedule
diffusion:
  schedule: "cosine"  # Change noise schedule
```

### Step 3: Train
```bash
python train.py --config tasks/my_experiment/config.yaml --exp_dir tasks/my_experiment
```

### Step 4: Infer
```bash
python inference.py --config tasks/my_experiment/config.yaml --exp_dir tasks/my_experiment
```

## 🎯 Experiment Organization

Each experiment has its own directory:
```
tasks/my_experiment/
├── config.yaml        # Configuration
├── checkpoints/       # Saved models
│   ├── checkpoint_best.pt
│   └── checkpoint_last.pt
├── logs/              # Training logs
└── figs/              # Generated figures
    ├── samples_cfg2.0.png
    └── training_history.png
```

## 🔧 Configuration Examples

### GPU Memory Issues
```yaml
training:
  batch_size: 1       # Reduce from 2
model:
  base_channels: 32    # Reduce from 64
```

### Want Better Quality?
```yaml
model:
  base_channels: 128   # Larger model
training:
  num_epochs: 500     # Train longer
```

### Want Faster Training?
```yaml
sampling:
  ddim_timesteps: 20  # Fewer sampling steps
training:
  batch_size: 8       # Larger batches
```

## 🆕 Adding New Features

### Add a New Noise Schedule
Edit `diffusion/schedules.py`:
```python
class MySchedule(NoiseSchedule):
    def get_betas(self):
        # Your implementation
        pass
```

Then in config:
```yaml
diffusion:
  schedule: "my_schedule"
```

### Add a New Model
Create `models/my_model.py`:
```python
class MyModel(nn.Module):
    def __init__(self, ...):
        # Your architecture
        pass
```

Then in `models/__init__.py`:
```python
from .my_model import MyModel
__all__ = [..., 'MyModel']
```

### Add Flow Matching
Create `flowmatching/flow_matching.py`:
```python
class FlowMatching(nn.Module):
    # Your implementation
    pass
```

Then use it in training scripts.

## 📊 Monitoring Training

Check training progress:
```bash
# View latest checkpoint
ls -lh tasks/my_experiment/checkpoints/

# View latest samples
ls -lh tasks/my_experiment/figs/
```

Training history is saved as `training_history.png` showing:
- Train/Val loss curves
- Learning rate schedule

## 🔍 Common Issues

### Import Error
```python
# Add to your script:
import sys
sys.path.append('/path/to/cosmo')
```

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `base_channels`
- Use gradient checkpointing

### Training Diverges
- Reduce learning rate
- Increase gradient clipping
- Check data normalization

## 📚 Next Steps

1. **Try different schedules**: Edit `diffusion/schedules.py`
2. **Experiment with CFG**: Change `cfg_prob` and `cfg_scale`
3. **Add new models**: Create in `models/`
4. **Implement flow matching**: Add to `flowmatching/`
5. **Parameter inference**: Extend `parameter_inference/`

Happy experimenting! 🚀

