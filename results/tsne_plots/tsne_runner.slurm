#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ee638

# run
echo "RUNNING_TSNE"
python3 tsne_patches.py
echo "DONE"
