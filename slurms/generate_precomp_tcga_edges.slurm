#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "EE-638-WSI-Project/slurms" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "EE-638-WSI-Project/slurms" FOR THIS PROJECT'
    return -1
fi

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ee638

# run
echo ""
echo "GENERATING GRAPH EDGES FOR TCGA PRECOMPUTED FEATURES"
echo ""

python3 get_edges.py --dataset=TCGA-lung-default
