#!/bin/bash
#SBATCH --gres=gpu:1                  # Number of GPUs (per node)
#SBATCH --mem=4000M                   # memory (per node)
#SBATCH --time=0-00:10                # time (DD-HH:MM)
#SBATCH --output=slurm-%j.out         # output filename pattern; j == jobid
source activate pytorch4
python main.py --no-progress
