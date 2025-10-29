# Flow Matching for FOCUS

Flow Matching implementation for cosmological field generation.

## Overview

Flow Matching is an alternative to Diffusion models that offers:
- **Simpler training objective**: Direct vector field prediction
- **Faster sampling**: Efficient ODE integration with fewer steps
- **Better sample quality**: Straight-line paths in latent space
- **Continuous time formulation**: No discrete timesteps

## Key Differences from Diffusion

| Aspect | Diffusion | Flow Matching |
|--------|-----------|---------------|
| **Training** | Predict noise ε | Predict vector field v |
| **Time** | Discrete timesteps (T=1000) | Continuous time t ∈ [0,1] |
| **Paths** | Random noise process | Straight-line interpolation |
| **Sampling** | DDPM/DDIM (50-1000 steps) | ODE (10-50 steps) |
| **Loss** | MSE(ε_pred, ε_true) | MSE(v_pred, v_true) |

## Usage

### 1. Configuration

Set `method: "flow"` in your YAML config:

```yaml
# Method selection
method: "flow"  # Use Flow Matching

# Model (use FlowUNet)
model:
  from: "FlowUNet"
  base_channels: 128
  time_dim: 256

# Flow Matching settings
flow_matching:
  sigma_min: 0.0
  sigma_max: 1.0

# Sampling
sampling:
  method: "euler"  # or "heun"
  num_steps: 50
  cfg_scale: 2.0
```

### 2. Training

```bash
cd /home/work/Cosmology/FOCUS
python train.py --config configs/example_flow.yaml --exp_dir experiments/flow_01
```

### 3. Switching Between Methods

Simply change the `method` field in your config:

```yaml
# Use Flow Matching
method: "flow"
model:
  from: "FlowUNet"

# Or use Diffusion (default)
method: "diffusion"
model:
  from: "SimpleUNet"
diffusion:
  timesteps: 1000
  ...
```

## Files

- `flow_matching.py` - Core Flow Matching implementation
- `flow_trainer.py` - Training loop (compatible with DiffusionTrainer)
- `flow_model.py` - FlowUNet model architecture
- `README.md` - This file

## Model Architecture

`FlowUNet` extends `SimpleUNet` with:
- Time embedding for continuous t ∈ [0, 1]
- Vector field prediction (instead of noise)
- Optimized initialization for stability

## Sampling Methods

### Euler Method (faster, 1st order)
```python
sampling:
  method: "euler"
  num_steps: 50
```

### Heun Method (better quality, 2nd order)
```python
sampling:
  method: "heun"
  num_steps: 25  # Can use fewer steps
```

## Theory

Flow Matching learns a time-dependent vector field v(x_t, t) that transports samples from noise to data:

```
dx/dt = v(x_t, t)
```

Training objective:
```
L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
```

where x_t = (1-t)x_0 + t·x_1

## References

- Lipman et al. (2023). "Flow Matching for Generative Modeling"
- Liu et al. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"

## Tips

1. **Start with Euler method** (simpler, faster)
2. **Use 50 steps** for good quality
3. **CFG scale 2.0** works well for cosmology
4. **Batch size** can be same as Diffusion
5. **Learning rate** can be same as Diffusion (1e-4)





