#!/bin/bash
#SBATCH --job-name=gp_env_gpu
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00
#SBATCH --output=/home/gdehol/logs/gp_env_gpu_%j.txt

# Build the CUDA env on a GPU node so the NVIDIA driver and CUDA
# libs are visible at install time.
source $HOME/data/miniforge3/etc/profile.d/conda.sh
mamba env create -f $HOME/git/gp_prior_fmri/create_env/environment_cuda.yml
