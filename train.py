"""
Main training script - uses experiment configs.
"""
import sys
import argparse

from tasks.train_experiment import main as train_main

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Usage: python train.py --config <config.yaml> --exp_dir <exp_dir>")
        print("\nExample:")
        print("  python train.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01")
        sys.exit(1)
    
    train_main()
