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


echo "6g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_41.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_41.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_42.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_42.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_43.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_43.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_44.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_44.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_45.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_45.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_46.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_46.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_47.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_47.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_48.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_48.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_49.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_49.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_50.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_50.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_51.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_51.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_52.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_52.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_53.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_53.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_54.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_54.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_55.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_55.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_56.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_56.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_57.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_57.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_58.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_58.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_59.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_59.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_60.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_60.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_61.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_61.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_62.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_62.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_63.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_63.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_64.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_64.tif"

echo "COMPLETE"
