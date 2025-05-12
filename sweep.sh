#!/bin/bash

#SBATCH --job-name=cecgnet
#SBATCH -c 8
#SBATCH --gres=gpu:3
#SBATCH -o current_run.txt
#SBATCH --ntasks-per-node=3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cecgnet

# Run the Python script
srun wandb agent tal4tal4-technion-israel-institute-of-technology/cecgnet/um3bxs8u
# please notice the manual config must be false for sweep to work
