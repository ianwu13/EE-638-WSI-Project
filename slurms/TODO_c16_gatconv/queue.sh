cd 2_edges
sbatch 1_layers.slurm
sbatch 2_layers.slurm
sbatch 3_layers.slurm
cd ..

cd 4_edges
sbatch 1_layers.slurm
sbatch 2_layers.slurm
sbatch 3_layers.slurm
cd ..

cd 8_edges
sbatch 1_layers.slurm
sbatch 2_layers.slurm
sbatch 3_layers.slurm
cd ..

cd 16_edges
sbatch 1_layers.slurm
sbatch 2_layers.slurm
sbatch 3_layers.slurm
cd ..

cd 32_edges
sbatch 1_layers.slurm
sbatch 2_layers.slurm
sbatch 3_layers.slurm
cd ..
