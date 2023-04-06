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


echo "5g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_17.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_17.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_18.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_18.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_19.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_19.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_20.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_20.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_21.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_21.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_22.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_22.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_23.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_23.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_24.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_24.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_25.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_25.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_26.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_26.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_27.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_27.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_28.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_28.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_29.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_29.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_30.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_30.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_31.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_31.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_32.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_32.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_33.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_33.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_34.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_34.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_35.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_35.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_36.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_36.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_37.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_37.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_38.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_38.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_39.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_39.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_40.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_40.tif"

echo "COMPLETE"
