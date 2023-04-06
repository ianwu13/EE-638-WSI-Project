#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "slurm_getters" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "slurm_getters" DIRECTORY FOR THIS PROJECT'
    return -1
fi


echo "10g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_137.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_137.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_138.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_138.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_139.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_139.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_140.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_140.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_141.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_141.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_142.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_142.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_143.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_143.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_144.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_144.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_145.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_145.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_146.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_146.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_147.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_147.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_148.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_148.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_149.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_149.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_150.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_150.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_151.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_151.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_152.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_152.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_153.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_153.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_154.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_154.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_155.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_155.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_156.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_156.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_157.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_157.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_158.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_158.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_159.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_159.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_160.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_160.tif"

echo "COMPLETE"
