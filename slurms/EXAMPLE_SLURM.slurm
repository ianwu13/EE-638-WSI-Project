#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00


# oTHER EXAMPLE ARGS
# OUT.write("#!/bin/bash\n")
# OUT.write("#SBATCH --job-name={} \n".format(job_name))
# OUT.write("#SBATCH --nodes=1 \n")
# OUT.write("#SBATCH --ntasks=1 \n")
# OUT.write("#SBATCH --cpus-per-task=16 \n")
# OUT.write("##SBATCH --time=47:59:59 \n")
# OUT.write("#SBATCH --account=ajiteshs_1045\n")
# OUT.write("#SBATCH --output={}/{}.out \n".format(LOG_DIR, job_name))
# OUT.write("#SBATCH --error={}/{}.out \n".format(ERROR_DIR, job_name))


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
echo "INFO HERE"
python3 SCRIPT_NAME_HERE.py
