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


echo "8g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_89.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_89.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_90.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_90.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_91.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_91.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_92.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_92.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_93.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_93.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_94.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_94.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_95.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_95.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_96.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_96.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_97.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_97.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_98.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_98.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_99.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_99.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_100.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_100.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_101.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_101.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_102.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_102.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_103.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_103.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_104.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_104.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_105.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_105.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_106.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_106.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_107.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_107.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_108.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_108.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_109.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_109.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_110.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_110.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_111.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_111.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_112.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_112.tif"

echo "COMPLETE"
