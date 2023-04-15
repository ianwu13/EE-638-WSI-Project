#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

# Change to proper directory
cd ../../..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ee638

# run
echo ""
echo "*-*-ID-*-* TRAINING GCN AGG W 2 LAYERS, 8 EDGES PER NODE ON TCGA-lung-default"
echo ""

# TCGA-lung-default
# Camelyon16
python3 train_gcn_agg.py --dataset=TCGA-lung-default --num_epochs=100 --model=graph_dsmil --edges_per_node=8 --gcn_layer_type=GraphConv --n_gcn_layers=2 --agg_type=dsmil

echo "DONE"
