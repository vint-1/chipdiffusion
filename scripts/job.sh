#! /bin/bash

#SBATCH --job-name=train
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --output=/global/scratch/users/%u/chipdiffusion/logs/%j.out

eval "$(conda shell.bash hook)"
conda activate chipdiffusion
PYTHONPATH=. python diffusion/train_graph.py $*