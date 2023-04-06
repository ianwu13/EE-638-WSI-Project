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


echo "4g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_123.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_123.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_124.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_124.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_125.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_125.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_126.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_126.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_127.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_127.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_128.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_128.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_129.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_129.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_130.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_130.tif"
# TRAINING
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_1.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_1.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_2.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_2.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_3.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_3.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_4.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_4.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_5.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_5.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_6.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_6.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_7.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_7.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_8.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_8.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_9.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_9.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_10.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_10.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_11.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_11.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_12.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_12.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_13.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_13.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_14.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_14.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_15.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_15.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/normal_16.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/normal/normal_16.tif"

echo "COMPLETE"
