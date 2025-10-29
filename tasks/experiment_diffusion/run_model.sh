#!/bin/bash

# Diffusion Experiment Runner
set -euo pipefail

# Defaults
CONFIG=${CONFIG:-"./config.yaml"}
EXP_DIR=${EXP_DIR:-"$(pwd)"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"../../train.py"}
PYTHON=${PYTHON:-"python3"}

echo "=========================================="
echo "Diffusion Experiment"
echo "=========================================="
echo "Config       : $CONFIG"
echo "Exp Dir      : $EXP_DIR"
echo "Train Script : $TRAIN_SCRIPT"
echo "Python       : $PYTHON"
echo "=========================================="

# Ensure output directories exist
mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/figs"

# Run training
"$PYTHON" "$TRAIN_SCRIPT" --config "$CONFIG" --exp_dir "$EXP_DIR"

echo "=========================================="
echo "Training completed"
echo "Results saved to: $EXP_DIR"
echo "=========================================="





