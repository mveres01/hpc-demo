#!/bin/bash
#SBATCH --account=def-mmoussa
#SBATCH --gres=gpu:1                  # Number of GPUs (per node)
#SBATCH --mem=4000M                   # memory (per node)
#SBATCH --time=0-00:10                # time (DD-HH:MM)
#SBATCH --output=slurm-%j.out         # output filename pattern; j == jobid
#SBATCH --reservation=guelph_workshop
module load miniconda3
source activate pytorch4
python main.py --no-progress
