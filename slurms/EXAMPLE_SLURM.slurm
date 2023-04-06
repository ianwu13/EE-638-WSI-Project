#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "EE-638-WSI-Project/slurms" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "EE-638-WSI-Project/slurms" FOR THIS PROJECT'
    return -1
fi

# load modules
# module load python/3.7.6
# module load git/2.25.0

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# load libraries
# pip install -e .

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ee638

# run
echo "TRAINING T5 NEXT UTTERANCE MODEL - INPUT: NOP"
python3 SCRIPT_NAME_HERE.py
