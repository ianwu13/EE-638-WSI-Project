#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

# Change to proper directory
if [[ "$PWD" =~ "EE-638-WSI-Project/slurms/c16_gcn" ]]
then 
    cd ../..
else
    echo 'SLURM JOBS MUST BE RUN FROM "EE-638-WSI-Project/slurms/c16_gcn" FOR THIS JOB'
    return -1
fi

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ee638

# run
echo ""
echo "TRAINING GCN AGG W 1 LAYERS ON Camelyon16"
echo ""

# TCGA-lung-default
# Camelyon16
python3 train_gcn_agg.py --dataset=Camelyon16 --num_epochs=100 --model=graph_dsmil --gcn_layer_type=GraphConv --n_gcn_layers=1 --agg_type=dsmil

echo "DONE"
