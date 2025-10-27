"""
Main inference script - uses experiment configs.
"""
import sys
import argparse

from tasks.inference_experiment import main as inference_main

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Usage: python inference.py --config <config.yaml> --exp_dir <exp_dir>")
        print("\nExample:")
        print("  python inference.py --config tasks/experiment_01/config.yaml --exp_dir tasks/experiment_01")
        sys.exit(1)
    
    inference_main()
