#!/bin/bash

# Minimal experiment run script
# Configure via env vars or flags. No automatic project-root detection.
# Defaults:
#   CONFIG=./config.yaml
#   EXP_DIR=$(pwd)
#   TRAIN_SCRIPT=../../train.py   # relative to this experiment folder
#   PYTHON=python3

set -euo pipefail

# Defaults (can be overridden by env or flags)
CONFIG=${CONFIG:-"./config.yaml"}
EXP_DIR=${EXP_DIR:-"$(pwd)"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"../../train.py"}
PYTHON=${PYTHON:-"python3"}

usage() {
  echo "Usage: $0 [--config PATH] [--exp_dir PATH] [--train PATH] [--python CMD]"
  echo "  --config PATH   Path to config.yaml   (default: ./config.yaml)"
  echo "  --exp_dir PATH  Experiment output dir (default: current directory)"
  echo "  --train PATH    Path to train.py      (default: ../../train.py)"
  echo "  --python CMD    Python interpreter    (default: python3)"
}

# Parse flags (simple parser)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2;;
    --exp_dir)
      EXP_DIR="$2"; shift 2;;
    --train)
      TRAIN_SCRIPT="$2"; shift 2;;
    --python)
      PYTHON="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"; usage; exit 1;;
  esac
done

echo "=========================================="
echo "Config       : $CONFIG"
echo "Exp Dir      : $EXP_DIR"
echo "Train Script : $TRAIN_SCRIPT"
echo "Python       : $PYTHON"
echo "=========================================="

# Ensure output directories exist
mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/figs"

# Run training (no directory changes)
"$PYTHON" "$TRAIN_SCRIPT" --config "$CONFIG" --exp_dir "$EXP_DIR"

echo "=========================================="
echo "Training completed"
echo "Results saved to: $EXP_DIR"
echo "=========================================="
