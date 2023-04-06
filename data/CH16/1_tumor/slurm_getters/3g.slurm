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


echo "3g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_122.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_122.tif"
# TRAINING
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_1.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_1.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_2.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_2.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_3.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_3.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_4.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_4.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_5.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_5.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_6.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_6.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_7.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_7.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_8.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_8.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_9.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_9.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_10.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_10.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_11.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_11.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_12.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_12.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_13.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_13.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_14.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_14.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_15.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_15.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_16.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_16.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_17.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_17.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_18.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_18.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_19.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_19.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_20.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_20.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_21.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_21.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_22.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_22.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_23.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_23.tif"

echo "COMPLETE"
