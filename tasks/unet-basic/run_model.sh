#!/bin/sh
hostname
source ~/.bashrc
micromamba activate gh-separation-test
cd ~
pwd
cd cosmology/COSMO
pwd
nvidia-smi
python3 train.py
