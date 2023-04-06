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


echo "9g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_113.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_113.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_114.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_114.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_115.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_115.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_116.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_116.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_117.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_117.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_118.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_118.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_119.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_119.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_120.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_120.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_121.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_121.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_122.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_122.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_123.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_123.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_124.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_124.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_125.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_125.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_126.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_126.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_127.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_127.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_128.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_128.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_129.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_129.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_130.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_130.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_131.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_131.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_132.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_132.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_133.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_133.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_134.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_134.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_135.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_135.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_136.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_136.tif"

echo "COMPLETE"
