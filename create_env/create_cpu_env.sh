#!/bin/bash
#SBATCH --job-name=gp_env_cpu
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=60:00
#SBATCH --output=/home/gdehol/logs/gp_env_cpu_%j.txt

source $HOME/data/miniforge3/etc/profile.d/conda.sh
mamba env create -f $HOME/git/gp_prior_fmri/create_env/environment_cpu.yml
