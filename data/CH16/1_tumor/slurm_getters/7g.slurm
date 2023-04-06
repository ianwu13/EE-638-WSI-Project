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


echo "7g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_96.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_96.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_97.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_97.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_98.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_98.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_99.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_99.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_100.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_100.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_101.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_101.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_102.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_102.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_103.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_103.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_104.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_104.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_105.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_105.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_106.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_106.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_107.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_107.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_108.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_108.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_109.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_109.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_110.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_110.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_111.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_111.tif"

echo "COMPLETE"
