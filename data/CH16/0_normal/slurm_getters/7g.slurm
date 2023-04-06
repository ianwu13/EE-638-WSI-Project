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

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_65.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_65.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_66.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_66.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_67.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_67.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_68.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_68.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_69.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_69.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_70.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_70.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_71.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_71.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_72.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_72.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_73.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_73.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_74.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_74.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_75.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_75.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_76.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_76.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_77.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_77.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_78.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_78.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_79.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_79.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_80.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_80.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_81.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_81.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_82.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_82.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_83.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_83.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_84.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_84.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_85.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_85.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_86.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_86.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_87.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_87.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_88.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_88.tif"

echo "COMPLETE"
