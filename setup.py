"""
Setup script for FOCUS package.
"""
from setuptools import setup, find_packages

setup(
    name="focus",
    version="0.1.0",
    description="Flow Matching & Diffusion Model for Cosmological Universe Simulation",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)

